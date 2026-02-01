import { redirect } from '@sveltejs/kit';
import type { Actions } from './$types.js';
import { deleteSession } from '$lib/server/auth.js';

export const actions: Actions = {
	default: async ({ cookies }) => {
		const token = cookies.get('session');
		if (token) {
			deleteSession(token);
			cookies.delete('session', { path: '/' });
		}
		redirect(302, '/auth/login');
	}
};
