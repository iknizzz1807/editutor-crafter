import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types.js';
import { db } from '$lib/server/db/index.js';
import {
	domains,
	projects,
	milestones,
	userProgress,
	projectPrerequisites,
	projectResources,
	projectLanguages,
	learningOutcomes,
	projectTags
} from '$lib/server/db/schema.js';
import { eq, sql, count, and } from 'drizzle-orm';

export const load: PageServerLoad = async ({ params, locals }) => {
	const domain = db
		.select()
		.from(domains)
		.where(eq(domains.slug, params.domain))
		.get();

	if (!domain) error(404, 'Domain not found');

	const allProjects = db
		.select()
		.from(projects)
		.where(eq(projects.domainId, domain.id))
		.orderBy(projects.sortOrder)
		.all();

	// Get milestone counts per project
	const milestoneCounts = db
		.select({
			projectId: milestones.projectId,
			total: count(milestones.id)
		})
		.from(milestones)
		.where(
			sql`${milestones.projectId} IN (${sql.join(
				allProjects.map((p) => sql`${p.id}`),
				sql`, `
			)})`
		)
		.groupBy(milestones.projectId)
		.all();

	// Get completed counts for this user
	const completedCounts = locals.user
		? db
				.select({
					projectId: milestones.projectId,
					completed: count(userProgress.id)
				})
				.from(userProgress)
				.innerJoin(milestones, eq(userProgress.milestoneId, milestones.id))
				.where(
					and(
						eq(userProgress.userId, locals.user.id),
						eq(userProgress.status, 'completed'),
						sql`${milestones.projectId} IN (${sql.join(
							allProjects.map((p) => sql`${p.id}`),
							sql`, `
						)})`
					)
				)
				.groupBy(milestones.projectId)
				.all()
		: [];

	const msMap = new Map(milestoneCounts.map((m) => [m.projectId, m.total]));
	const completedMap = new Map(completedCounts.map((c) => [c.projectId, c.completed]));

	const projectsWithProgress = allProjects.map((p) => ({
		...p,
		totalMilestones: msMap.get(p.id) || 0,
		completedMilestones: completedMap.get(p.id) || 0
	}));

	// Group by difficulty
	const grouped = {
		beginner: projectsWithProgress.filter((p) => p.difficulty === 'beginner'),
		intermediate: projectsWithProgress.filter((p) => p.difficulty === 'intermediate'),
		advanced: projectsWithProgress.filter((p) => p.difficulty === 'advanced'),
		expert: projectsWithProgress.filter((p) => p.difficulty === 'expert')
	};

	return {
		domain,
		projects: grouped
	};
};
