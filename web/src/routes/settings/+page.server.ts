import { error, fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { userSettings } from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { encrypt, decrypt, maskApiKey } from '$lib/server/crypto.js';

export const load: PageServerLoad = async ({ locals }) => {
	if (!locals.user) error(401, 'Not authenticated');

	const settings = db
		.select()
		.from(userSettings)
		.where(eq(userSettings.userId, locals.user.id))
		.get();

	let maskedKey: string | null = null;
	if (settings?.geminiApiKey) {
		try {
			const decrypted = decrypt(settings.geminiApiKey);
			maskedKey = maskApiKey(decrypted);
		} catch {
			maskedKey = '****invalid****';
		}
	}

	return {
		hasApiKey: !!settings?.geminiApiKey,
		maskedApiKey: maskedKey
	};
};

export const actions: Actions = {
	saveApiKey: async ({ request, locals }) => {
		if (!locals.user) error(401, 'Not authenticated');

		const formData = await request.formData();
		const apiKey = (formData.get('apiKey') as string)?.trim();

		if (!apiKey) {
			return fail(400, { error: 'API key is required' });
		}

		const encrypted = encrypt(apiKey);

		const existing = db
			.select()
			.from(userSettings)
			.where(eq(userSettings.userId, locals.user.id))
			.get();

		if (existing) {
			db.update(userSettings)
				.set({
					geminiApiKey: encrypted,
					updatedAt: new Date().toISOString()
				})
				.where(eq(userSettings.id, existing.id))
				.run();
		} else {
			db.insert(userSettings)
				.values({
					userId: locals.user.id,
					geminiApiKey: encrypted
				})
				.run();
		}

		return { success: true, message: 'API key saved successfully' };
	},

	deleteApiKey: async ({ locals }) => {
		if (!locals.user) error(401, 'Not authenticated');

		const existing = db
			.select()
			.from(userSettings)
			.where(eq(userSettings.userId, locals.user.id))
			.get();

		if (existing) {
			db.update(userSettings)
				.set({
					geminiApiKey: null,
					updatedAt: new Date().toISOString()
				})
				.where(eq(userSettings.id, existing.id))
				.run();
		}

		return { success: true, message: 'API key deleted' };
	}
};
