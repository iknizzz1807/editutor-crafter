import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { submissions, aiInteractions, milestones, projects, domains } from '$lib/server/db/schema.js';
import { eq, desc, and, sql } from 'drizzle-orm';

export const GET: RequestHandler = async ({ locals, url }) => {
	if (!locals.user) error(401, 'Not authenticated');

	const page = parseInt(url.searchParams.get('page') || '1');
	const limit = parseInt(url.searchParams.get('limit') || '20');
	const offset = (page - 1) * limit;

	const allSubmissions = db
		.select({
			id: submissions.id,
			fileName: submissions.fileName,
			fileSize: submissions.fileSize,
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
		.limit(limit)
		.offset(offset)
		.all();

	// Get AI interactions for these submissions
	const submissionIds = allSubmissions.map((s) => s.id);
	const interactions =
		submissionIds.length > 0
			? db
					.select()
					.from(aiInteractions)
					.where(
						and(
							eq(aiInteractions.userId, locals.user.id),
							sql`${aiInteractions.submissionId} IN (${sql.join(
								submissionIds.map((id) => sql`${id}`),
								sql`, `
							)})`
						)
					)
					.all()
			: [];

	const interactionMap = new Map<number, typeof interactions>();
	for (const interaction of interactions) {
		if (interaction.submissionId) {
			const existing = interactionMap.get(interaction.submissionId) || [];
			existing.push(interaction);
			interactionMap.set(interaction.submissionId, existing);
		}
	}

	const result = allSubmissions.map((s) => ({
		...s,
		reviews: interactionMap.get(s.id) || []
	}));

	return json({ submissions: result, page, limit });
};
