import { redirect } from '@sveltejs/kit';
import type { LayoutServerLoad } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { domains, projects, milestones, userProgress } from '$lib/server/db/schema.js';
import { eq, sql, count } from 'drizzle-orm';

export const load: LayoutServerLoad = async ({ locals }) => {
	if (!locals.user) redirect(302, '/auth/login');

	const allDomains = db
		.select({
			id: domains.id,
			slug: domains.slug,
			name: domains.name,
			description: domains.description,
			icon: domains.icon,
			sortOrder: domains.sortOrder
		})
		.from(domains)
		.orderBy(domains.sortOrder)
		.all();

	const domainStats = db
		.select({
			domainId: projects.domainId,
			totalMilestones: count(milestones.id)
		})
		.from(milestones)
		.innerJoin(projects, eq(milestones.projectId, projects.id))
		.groupBy(projects.domainId)
		.all();

	const completedStats = db
		.select({
			domainId: projects.domainId,
			completedMilestones: count(userProgress.id)
		})
		.from(userProgress)
		.innerJoin(milestones, eq(userProgress.milestoneId, milestones.id))
		.innerJoin(projects, eq(milestones.projectId, projects.id))
		.where(
			sql`${userProgress.userId} = ${locals.user.id} AND ${userProgress.status} = 'completed'`
		)
		.groupBy(projects.domainId)
		.all();

	const statsMap = new Map(domainStats.map((s) => [s.domainId, s.totalMilestones]));
	const completedMap = new Map(completedStats.map((s) => [s.domainId, s.completedMilestones]));

	const domainsWithProgress = allDomains.map((d) => ({
		...d,
		totalMilestones: statsMap.get(d.id) || 0,
		completedMilestones: completedMap.get(d.id) || 0
	}));

	const totalMilestones = domainsWithProgress.reduce((sum, d) => sum + d.totalMilestones, 0);
	const totalCompleted = domainsWithProgress.reduce((sum, d) => sum + d.completedMilestones, 0);

	return {
		domains: domainsWithProgress,
		totalMilestones,
		totalCompleted,
		user: locals.user
	};
};
