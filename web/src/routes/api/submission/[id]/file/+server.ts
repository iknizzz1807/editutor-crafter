import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { submissions } from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { resolve } from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { getFileTree, getLanguage } from '$lib/server/tree-builder.js';

const execAsync = promisify(exec);

const MAX_FILE_SIZE = 100 * 1024; // 100KB

function isBinary(buffer: Buffer): boolean {
	const checkLength = Math.min(buffer.length, 512);
	for (let i = 0; i < checkLength; i++) {
		if (buffer[i] === 0) return true;
	}
	return false;
}

export const GET: RequestHandler = async ({ params, url, locals }) => {
	if (!locals.user) error(401, 'Not authenticated');

	const submissionId = parseInt(params.id, 10);
	if (isNaN(submissionId)) {
		return json({ error: 'Invalid submission ID' }, { status: 400 });
	}

	const filePath = url.searchParams.get('path');
	if (!filePath) {
		return json({ error: 'path parameter is required' }, { status: 400 });
	}

	// Security: reject path traversal
	if (filePath.includes('..') || filePath.startsWith('/')) {
		return json({ error: 'Invalid file path' }, { status: 400 });
	}

	const submission = db
		.select()
		.from(submissions)
		.where(eq(submissions.id, submissionId))
		.get();

	if (!submission || submission.userId !== locals.user.id) {
		return json({ error: 'Submission not found' }, { status: 404 });
	}

	const zipPath = resolve(`data/${submission.filePath}`);

	// Get tree to resolve rootPrefix
	let rootPrefix: string;
	try {
		const treeResult = await getFileTree(zipPath, submissionId);
		rootPrefix = treeResult.rootPrefix;
	} catch {
		return json({ error: 'Failed to read zip file' }, { status: 500 });
	}

	// Build the actual path inside the zip
	const actualPath = rootPrefix ? `${rootPrefix}/${filePath}` : filePath;

	// Get file size first by listing
	try {
		const { stdout: listOut } = await execAsync(
			`unzip -l "${zipPath}" "${actualPath}"`,
			{ maxBuffer: 1024 * 1024 }
		);

		// Parse size from listing
		const sizeMatch = listOut.match(new RegExp(`^\\s*(\\d+)\\s+.*${actualPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'm'));
		const fileSize = sizeMatch ? parseInt(sizeMatch[1], 10) : 0;

		const ext = filePath.includes('.') ? '.' + filePath.split('.').pop()! : '';
		const language = getLanguage(ext);

		if (fileSize > MAX_FILE_SIZE) {
			return json({
				path: filePath,
				size: fileSize,
				language,
				content: null,
				truncated: true,
				binary: false
			});
		}

		// Extract file content
		const { stdout } = await execAsync(
			`unzip -p "${zipPath}" "${actualPath}"`,
			{ maxBuffer: 1024 * 1024, encoding: 'buffer' as any }
		);

		const buffer = Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout as any);

		if (isBinary(buffer)) {
			return json({
				path: filePath,
				size: fileSize,
				language,
				content: null,
				truncated: false,
				binary: true
			});
		}

		const content = buffer.toString('utf8');

		return json({
			path: filePath,
			size: fileSize,
			language,
			content,
			truncated: false,
			binary: false
		});
	} catch {
		return json({ error: 'Failed to extract file' }, { status: 500 });
	}
};
