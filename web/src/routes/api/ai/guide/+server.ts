import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { milestones, aiInteractions, userSettings, projects } from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { decrypt } from '$lib/server/crypto.js';
import { guideStudent } from '$lib/server/gemini.js';

export const POST: RequestHandler = async ({ request, locals }) => {
	if (!locals.user) error(401, 'Not authenticated');

	const body = await request.json();
	const { milestoneId, question } = body;

	if (!milestoneId || !question) {
		return json({ error: 'milestoneId and question are required' }, { status: 400 });
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

	// Get milestone
	const milestone = db
		.select()
		.from(milestones)
		.where(eq(milestones.id, milestoneId))
		.get();

	if (!milestone) {
		return json({ error: 'Milestone not found' }, { status: 404 });
	}

	const project = db
		.select()
		.from(projects)
		.where(eq(projects.id, milestone.projectId))
		.get();

	try {
		const guidance = await guideStudent(apiKey, milestone, question);

		// Save AI interaction
		db.insert(aiInteractions)
			.values({
				userId: locals.user.id,
				projectId: project?.id || null,
				milestoneId: milestone.id,
				interactionType: 'guide',
				prompt: question,
				response: guidance
			})
			.run();

		return json({ guidance });
	} catch (err: any) {
		const message = err?.message || 'AI guidance failed';
		if (message.includes('API_KEY_INVALID') || message.includes('401')) {
			return json({ error: 'Invalid Gemini API key. Please check your Settings.' }, { status: 401 });
		}
		return json({ error: `AI guidance failed: ${message}` }, { status: 500 });
	}
};
