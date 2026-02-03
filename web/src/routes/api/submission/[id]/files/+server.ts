import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { submissions } from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { resolve } from 'path';
import { getFileTree } from '$lib/server/tree-builder.js';

export const GET: RequestHandler = async ({ params, locals }) => {
	if (!locals.user) error(401, 'Not authenticated');

	const submissionId = parseInt(params.id, 10);
	if (isNaN(submissionId)) {
		return json({ error: 'Invalid submission ID' }, { status: 400 });
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

	try {
		const result = await getFileTree(zipPath, submissionId);
		return json(result);
	} catch {
		return json({ error: 'Failed to read zip file' }, { status: 500 });
	}
};
