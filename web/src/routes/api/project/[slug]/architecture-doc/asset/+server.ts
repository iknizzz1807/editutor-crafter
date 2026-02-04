import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { projects } from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { readFileSync } from 'fs';
import { resolve, dirname, normalize } from 'path';

const ALLOWED_EXTENSIONS = new Set(['.svg', '.png', '.d2']);

const CONTENT_TYPES: Record<string, string> = {
	'.svg': 'image/svg+xml',
	'.png': 'image/png',
	'.d2': 'text/plain'
};

export const GET: RequestHandler = async ({ params, url }) => {
	const assetPath = url.searchParams.get('path');
	if (!assetPath) error(400, 'Missing path parameter');

	// Security: reject path traversal
	if (assetPath.includes('..') || assetPath.startsWith('/')) {
		error(400, 'Invalid path');
	}

	// Check extension
	const ext = '.' + assetPath.split('.').pop()?.toLowerCase();
	if (!ALLOWED_EXTENSIONS.has(ext)) {
		error(400, 'File type not allowed');
	}

	const project = db
		.select()
		.from(projects)
		.where(eq(projects.slug, params.slug))
		.get();

	if (!project) error(404, 'Project not found');
	if (!project.architectureDocPath) error(404, 'No architecture document');

	const docDir = dirname(resolve(`../data/${project.architectureDocPath}`));
	const fullPath = normalize(resolve(docDir, assetPath));

	// Ensure the resolved path is within the doc directory
	if (!fullPath.startsWith(docDir)) {
		error(400, 'Invalid path');
	}

	try {
		const content = readFileSync(fullPath);
		return new Response(content, {
			headers: {
				'Content-Type': CONTENT_TYPES[ext] || 'application/octet-stream',
				'Cache-Control': 'public, max-age=3600'
			}
		});
	} catch {
		error(404, 'Asset not found');
	}
};
