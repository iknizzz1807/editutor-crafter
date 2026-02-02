import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import {
	submissions,
	milestones,
	aiInteractions,
	userSettings,
	projects
} from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { decrypt } from '$lib/server/crypto.js';
import { reviewSubmission } from '$lib/server/gemini.js';
import { resolve } from 'path';

async function extractTextFilesFromZip(zipPath: string): Promise<string> {
	const { exec } = await import('child_process');
	const { promisify } = await import('util');
	const execAsync = promisify(exec);

	// List files in zip
	let fileList: string;
	try {
		const { stdout } = await execAsync(`unzip -l "${zipPath}" | tail -n +4 | head -n -2 | awk '{print $4}'`, { maxBuffer: 1024 * 1024 });
		fileList = stdout.trim();
	} catch {
		return '[Could not list zip contents]';
	}

	const files = fileList.split('\n').filter((f) => f && !f.endsWith('/'));

	// Filter to source code files only
	const sourceExtensions = [
		'.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.c', '.cpp', '.h', '.hpp',
		'.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
		'.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
		'.json', '.yaml', '.yml', '.toml', '.xml', '.sql',
		'.sh', '.bash', '.zsh', '.fish', '.ps1',
		'.md', '.txt', '.cfg', '.ini', '.env.example',
		'.dockerfile', '.gitignore', '.editorconfig'
	];

	const sourceFiles = files.filter((f) => {
		const lower = f.toLowerCase();
		const name = lower.split('/').pop() || '';
		return sourceExtensions.some((ext) => lower.endsWith(ext)) ||
			name === 'dockerfile' || name === 'makefile' || name === 'rakefile' ||
			name === 'gemfile' || name === 'pipfile';
	});

	if (sourceFiles.length === 0) {
		return '[No source code files found in zip]';
	}

	// Extract and concat source files (limit total to ~50KB to not overwhelm the LLM)
	let result = '';
	const maxTotal = 50 * 1024;

	for (const file of sourceFiles.slice(0, 50)) {
		if (result.length >= maxTotal) {
			result += `\n\n... (truncated, ${sourceFiles.length - sourceFiles.indexOf(file)} more files)\n`;
			break;
		}

		try {
			const { stdout } = await execAsync(`unzip -p "${zipPath}" "${file}"`, {
				maxBuffer: 1024 * 1024,
				encoding: 'utf8'
			});
			const content = stdout.slice(0, 5000); // max 5KB per file
			result += `\n--- ${file} ---\n${content}\n`;
		} catch {
			result += `\n--- ${file} ---\n[Could not extract]\n`;
		}
	}

	return result || '[Empty zip]';
}

export const POST: RequestHandler = async ({ request, locals }) => {
	if (!locals.user) error(401, 'Not authenticated');

	const body = await request.json();
	const { submissionId } = body;

	if (!submissionId) {
		return json({ error: 'submissionId is required' }, { status: 400 });
	}

	// Get user's API key
	const settings = db
		.select()
		.from(userSettings)
		.where(eq(userSettings.userId, locals.user.id))
		.get();

	if (!settings?.geminiApiKey) {
		return json(
			{ error: 'No Gemini API key configured. Please add one in Settings.' },
			{ status: 400 }
		);
	}

	let apiKey: string;
	try {
		apiKey = decrypt(settings.geminiApiKey);
	} catch {
		return json({ error: 'Failed to decrypt API key. Please re-save it in Settings.' }, { status: 500 });
	}

	// Get submission
	const submission = db
		.select()
		.from(submissions)
		.where(eq(submissions.id, submissionId))
		.get();

	if (!submission || submission.userId !== locals.user.id) {
		return json({ error: 'Submission not found' }, { status: 404 });
	}

	// Get milestone with project info
	const milestone = db
		.select()
		.from(milestones)
		.where(eq(milestones.id, submission.milestoneId))
		.get();

	if (!milestone) {
		return json({ error: 'Milestone not found' }, { status: 404 });
	}

	const project = db
		.select()
		.from(projects)
		.where(eq(projects.id, milestone.projectId))
		.get();

	// Extract source code from zip
	const zipPath = resolve(`data/${submission.filePath}`);
	let sourceCode: string;
	try {
		sourceCode = await extractTextFilesFromZip(zipPath);
	} catch {
		return json({ error: 'Failed to read submission file' }, { status: 500 });
	}

	try {
		const review = await reviewSubmission(apiKey, milestone, sourceCode, null);

		// Save AI interaction
		db.insert(aiInteractions)
			.values({
				userId: locals.user.id,
				projectId: project?.id || null,
				milestoneId: milestone.id,
				submissionId: submission.id,
				interactionType: 'review',
				prompt: `Review submission for milestone: ${milestone.title}`,
				response: review
			})
			.run();

		// Update submission status
		db.update(submissions)
			.set({ status: 'reviewed' })
			.where(eq(submissions.id, submissionId))
			.run();

		return json({ review });
	} catch (err: any) {
		const message = err?.message || 'AI review failed';
		if (message.includes('API_KEY_INVALID') || message.includes('401')) {
			return json({ error: 'Invalid Gemini API key. Please check your Settings.' }, { status: 401 });
		}
		return json({ error: `AI review failed: ${message}` }, { status: 500 });
	}
};
