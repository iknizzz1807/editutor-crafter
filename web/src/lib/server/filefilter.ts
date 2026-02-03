const EXCLUDED_DIRS = new Set([
	'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env',
	'build', 'dist', 'target', 'out', '.next', '.nuxt', '.svelte-kit',
	'.cache', 'coverage', '.idea', '.vscode', '__MACOSX', 'vendor',
	'bin', 'obj', '.gradle', '.eggs', '.tox', '.DS_Store'
]);

const EXCLUDED_FILES = new Set([
	'.DS_Store', 'Thumbs.db', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml'
]);

export function shouldExcludePath(segments: string[]): boolean {
	return segments.some((seg) => EXCLUDED_DIRS.has(seg));
}

export function shouldExcludeFile(name: string): boolean {
	return EXCLUDED_FILES.has(name);
}
