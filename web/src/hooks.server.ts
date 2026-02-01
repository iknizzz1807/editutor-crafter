import type { Handle } from '@sveltejs/kit';
import { validateSession } from '$lib/server/auth.js';

export const handle: Handle = async ({ event, resolve }) => {
	const token = event.cookies.get('session');

	if (token) {
		const user = validateSession(token);
		event.locals.user = user;
	} else {
		event.locals.user = null;
	}

	return resolve(event);
};
