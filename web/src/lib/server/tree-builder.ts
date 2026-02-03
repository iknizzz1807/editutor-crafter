import { exec } from 'child_process';
import { promisify } from 'util';
import type { FileTreeNode } from '$lib/types.js';
import { shouldExcludePath, shouldExcludeFile } from './filefilter.js';

const execAsync = promisify(exec);

const LANGUAGE_MAP: Record<string, string> = {
	'.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
	'.ts': 'typescript', '.tsx': 'typescript', '.mts': 'typescript',
	'.py': 'python', '.pyw': 'python',
	'.java': 'java', '.kt': 'kotlin', '.kts': 'kotlin', '.scala': 'scala',
	'.c': 'c', '.h': 'c',
	'.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
	'.cs': 'csharp',
	'.go': 'go',
	'.rs': 'rust',
	'.rb': 'ruby',
	'.php': 'php',
	'.swift': 'swift',
	'.r': 'r', '.R': 'r',
	'.html': 'html', '.htm': 'html',
	'.css': 'css', '.scss': 'scss', '.sass': 'sass', '.less': 'less',
	'.vue': 'html', '.svelte': 'html',
	'.json': 'json',
	'.yaml': 'yaml', '.yml': 'yaml',
	'.toml': 'toml',
	'.xml': 'xml', '.svg': 'xml',
	'.sql': 'sql',
	'.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
	'.ps1': 'powershell',
	'.md': 'markdown', '.mdx': 'markdown',
	'.txt': 'plaintext',
	'.dockerfile': 'dockerfile',
	'.lua': 'lua',
	'.dart': 'dart',
	'.ex': 'elixir', '.exs': 'elixir',
	'.hs': 'haskell',
	'.ml': 'ocaml',
	'.clj': 'clojure',
	'.erl': 'erlang',
	'.zig': 'zig',
	'.nim': 'nim',
	'.v': 'v',
	'.pl': 'perl',
	'.m': 'objectivec',
	'.gradle': 'groovy',
	'.tf': 'hcl',
};

export function getLanguage(ext: string): string {
	return LANGUAGE_MAP[ext.toLowerCase()] || 'plaintext';
}

interface ZipEntry {
	size: number;
	path: string;
}

interface CachedTree {
	tree: FileTreeNode;
	rootPrefix: string;
	fileCount: number;
	totalSize: number;
}

const treeCache = new Map<number, CachedTree>();

function parseUnzipList(output: string): ZipEntry[] {
	const entries: ZipEntry[] = [];
	const lines = output.split('\n');

	for (const line of lines) {
		// unzip -l format: "  Length      Date    Time    Name"
		// e.g. "     1234  2024-01-01 12:00   path/to/file.txt"
		const match = line.match(/^\s*(\d+)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+(.+)$/);
		if (match) {
			const size = parseInt(match[1], 10);
			const path = match[2];
			if (path && path !== '') {
				entries.push({ size, path });
			}
		}
	}

	return entries;
}

function buildTree(entries: ZipEntry[]): { tree: FileTreeNode; rootPrefix: string; fileCount: number; totalSize: number } {
	const root: FileTreeNode = { name: '', path: '', type: 'directory', children: [] };
	let fileCount = 0;
	let totalSize = 0;

	for (const entry of entries) {
		const segments = entry.path.split('/').filter(Boolean);
		const isDir = entry.path.endsWith('/');

		if (shouldExcludePath(segments)) continue;

		if (isDir) {
			// Ensure directory nodes exist
			let current = root;
			let currentPath = '';
			for (const seg of segments) {
				currentPath = currentPath ? `${currentPath}/${seg}` : seg;
				let child = current.children!.find((c) => c.name === seg && c.type === 'directory');
				if (!child) {
					child = { name: seg, path: currentPath, type: 'directory', children: [] };
					current.children!.push(child);
				}
				current = child;
			}
		} else {
			const fileName = segments[segments.length - 1];
			if (shouldExcludeFile(fileName)) continue;

			// Ensure parent directories exist
			let current = root;
			let currentPath = '';
			for (let i = 0; i < segments.length - 1; i++) {
				currentPath = currentPath ? `${currentPath}/${segments[i]}` : segments[i];
				let child = current.children!.find((c) => c.name === segments[i] && c.type === 'directory');
				if (!child) {
					child = { name: segments[i], path: currentPath, type: 'directory', children: [] };
					current.children!.push(child);
				}
				current = child;
			}

			const ext = fileName.includes('.') ? '.' + fileName.split('.').pop()! : '';
			current.children!.push({
				name: fileName,
				path: segments.join('/'),
				type: 'file',
				size: entry.size,
				extension: ext || undefined
			});

			fileCount++;
			totalSize += entry.size;
		}
	}

	// Sort recursively: directories first, then alphabetical
	sortTree(root);

	// Collapse single-child root
	let resultTree = root;
	let rootPrefix = '';

	while (
		resultTree.children &&
		resultTree.children.length === 1 &&
		resultTree.children[0].type === 'directory'
	) {
		const child = resultTree.children[0];
		rootPrefix = child.path;
		resultTree = child;
	}

	// Reset the unwrapped root's path to empty (it becomes the new root)
	resultTree.path = '';
	resultTree.name = resultTree.name || 'root';

	return { tree: resultTree, rootPrefix, fileCount, totalSize };
}

function sortTree(node: FileTreeNode) {
	if (!node.children) return;

	node.children.sort((a, b) => {
		if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
		return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
	});

	for (const child of node.children) {
		if (child.type === 'directory') sortTree(child);
	}
}

export async function getFileTree(zipPath: string, submissionId: number): Promise<CachedTree> {
	const cached = treeCache.get(submissionId);
	if (cached) return cached;

	const { stdout } = await execAsync(`unzip -l "${zipPath}"`, { maxBuffer: 2 * 1024 * 1024 });
	const entries = parseUnzipList(stdout);

	if (entries.length === 0) {
		const empty: CachedTree = {
			tree: { name: 'root', path: '', type: 'directory', children: [] },
			rootPrefix: '',
			fileCount: 0,
			totalSize: 0
		};
		treeCache.set(submissionId, empty);
		return empty;
	}

	const result = buildTree(entries);
	treeCache.set(submissionId, result);
	return result;
}
