import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { submissions, aiInteractions, milestones, projects, domains } from '$lib/server/db/schema.js';
import { eq, desc, and } from 'drizzle-orm';

export const load: PageServerLoad = async ({ locals }) => {
	if (!locals.user) error(401, 'Not authenticated');

	const allSubmissions = db
		.select({
			id: submissions.id,
			content: submissions.content,
			language: submissions.language,
			status: submissions.status,
			createdAt: submissions.createdAt,
			milestoneId: submissions.milestoneId,
			milestoneTitle: milestones.title,
			projectName: projects.name,
			projectSlug: projects.slug,
			domainName: domains.name,
			domainSlug: domains.slug,
			domainIcon: domains.icon
		})
		.from(submissions)
		.innerJoin(milestones, eq(submissions.milestoneId, milestones.id))
		.innerJoin(projects, eq(milestones.projectId, projects.id))
		.innerJoin(domains, eq(projects.domainId, domains.id))
		.where(eq(submissions.userId, locals.user.id))
		.orderBy(desc(submissions.createdAt))
		.all();

	// Get reviews for submissions
	const reviews = db
		.select()
		.from(aiInteractions)
		.where(
			and(
				eq(aiInteractions.userId, locals.user.id),
				eq(aiInteractions.interactionType, 'review')
			)
		)
		.orderBy(desc(aiInteractions.createdAt))
		.all();

	const reviewsMap = new Map<number, string>();
	for (const r of reviews) {
		if (r.submissionId && !reviewsMap.has(r.submissionId)) {
			reviewsMap.set(r.submissionId, r.response || '');
		}
	}

	const result = allSubmissions.map((s) => ({
		...s,
		review: reviewsMap.get(s.id) || null
	}));

	return { submissions: result };
};
