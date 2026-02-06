import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { projects } from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { renderArchDocCached } from '$lib/markdown.js';

export const GET: RequestHandler = async ({ params }) => {
	const project = db
		.select()
		.from(projects)
		.where(eq(projects.slug, params.slug))
		.get();

	if (!project) error(404, 'Project not found');
	if (!project.architectureDocPath) error(404, 'No architecture document for this project');

	// Try pre-generated file first
	const generatedPath = resolve('../data/generated-docs', params.slug, 'doc.json');
	if (existsSync(generatedPath)) {
		try {
			const raw = readFileSync(generatedPath, 'utf-8');
			const doc = JSON.parse(raw);
			return json(doc);
		} catch {
			// Fall through to on-the-fly rendering
		}
	}

	// Fallback: render on-the-fly (slow for large docs)
	const docPath = resolve(`../data/${project.architectureDocPath}`);

	let markdown: string;
	try {
		markdown = readFileSync(docPath, 'utf-8');
	} catch {
		error(404, 'Architecture document file not found');
	}

	const assetBaseUrl = `/api/project/${params.slug}/architecture-doc/asset`;

	// Extract title from first H1
	const titleMatch = markdown.match(/^#\s+(.+)$/m);
	const title = titleMatch ? titleMatch[1] : project.name;

	const { html, toc } = await renderArchDocCached(
		project.slug,
		markdown,
		assetBaseUrl
	);

	return json({ html, markdown, title, toc });
};
