import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types.js';
import { findUserByUsername, verifyPassword, createSession } from '$lib/server/auth.js';
import { z } from 'zod';

const loginSchema = z.object({
	username: z.string().min(1, 'Username is required'),
	password: z.string().min(1, 'Password is required')
});

export const load: PageServerLoad = async ({ locals }) => {
	if (locals.user) redirect(302, '/roadmap');
};

export const actions: Actions = {
	default: async ({ request, cookies }) => {
		const formData = await request.formData();
		const data = {
			username: formData.get('username') as string,
			password: formData.get('password') as string
		};

		const parsed = loginSchema.safeParse(data);
		if (!parsed.success) {
			return fail(400, { error: parsed.error.errors[0].message, username: data.username });
		}

		const user = findUserByUsername(parsed.data.username);
		if (!user) {
			return fail(400, { error: 'Invalid username or password', username: data.username });
		}

		const valid = await verifyPassword(parsed.data.password, user.passwordHash);
		if (!valid) {
			return fail(400, { error: 'Invalid username or password', username: data.username });
		}

		const session = createSession(user.id);
		cookies.set('session', session.token, {
			path: '/',
			httpOnly: true,
			sameSite: 'lax',
			secure: false,
			maxAge: 30 * 24 * 60 * 60
		});

		redirect(302, '/roadmap');
	}
};
