import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types.js';
import {
	findUserByUsername,
	findUserByEmail,
	hashPassword,
	createUser,
	createSession
} from '$lib/server/auth.js';
import { z } from 'zod';

const registerSchema = z.object({
	username: z
		.string()
		.min(3, 'Username must be at least 3 characters')
		.max(30, 'Username must be at most 30 characters')
		.regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers, and underscores'),
	email: z.string().email('Invalid email address'),
	password: z.string().min(6, 'Password must be at least 6 characters'),
	confirmPassword: z.string()
}).refine((data) => data.password === data.confirmPassword, {
	message: 'Passwords do not match',
	path: ['confirmPassword']
});

export const load: PageServerLoad = async ({ locals }) => {
	if (locals.user) redirect(302, '/roadmap');
};

export const actions: Actions = {
	default: async ({ request, cookies }) => {
		const formData = await request.formData();
		const data = {
			username: formData.get('username') as string,
			email: formData.get('email') as string,
			password: formData.get('password') as string,
			confirmPassword: formData.get('confirmPassword') as string
		};

		const parsed = registerSchema.safeParse(data);
		if (!parsed.success) {
			return fail(400, {
				error: parsed.error.errors[0].message,
				username: data.username,
				email: data.email
			});
		}

		if (findUserByUsername(parsed.data.username)) {
			return fail(400, {
				error: 'Username already taken',
				username: data.username,
				email: data.email
			});
		}

		if (findUserByEmail(parsed.data.email)) {
			return fail(400, {
				error: 'Email already registered',
				username: data.username,
				email: data.email
			});
		}

		const passwordHash = await hashPassword(parsed.data.password);
		const user = createUser(parsed.data.username, parsed.data.email, passwordHash);
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
