import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';

export const POST: RequestHandler = async () => {
	return json(
		{ error: 'AI code review not yet implemented' },
		{ status: 501 }
	);
};
