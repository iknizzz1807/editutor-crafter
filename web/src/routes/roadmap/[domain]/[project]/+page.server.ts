import { error } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types.js';
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
import { eq, and } from 'drizzle-orm';

export const load: PageServerLoad = async ({ params, locals }) => {
	const domain = db.select().from(domains).where(eq(domains.slug, params.domain)).get();
	if (!domain) error(404, 'Domain not found');

	const project = db
		.select()
		.from(projects)
		.where(and(eq(projects.domainId, domain.id), eq(projects.slug, params.project)))
		.get();
	if (!project) error(404, 'Project not found');

	const allMilestones = db
		.select()
		.from(milestones)
		.where(eq(milestones.projectId, project.id))
		.orderBy(milestones.sortOrder)
		.all();

	// Get user progress for these milestones
	const progressData = locals.user
		? db
				.select()
				.from(userProgress)
				.where(eq(userProgress.userId, locals.user.id))
				.all()
		: [];

	const progressMap = new Map(progressData.map((p) => [p.milestoneId, p.status]));

	const milestonesWithProgress = allMilestones.map((m) => ({
		...m,
		status: progressMap.get(m.id) || 'not_started'
	}));

	// Load related data
	const prerequisites = db
		.select()
		.from(projectPrerequisites)
		.where(eq(projectPrerequisites.projectId, project.id))
		.all();

	const resources = db
		.select()
		.from(projectResources)
		.where(eq(projectResources.projectId, project.id))
		.all();

	const languages = db
		.select()
		.from(projectLanguages)
		.where(eq(projectLanguages.projectId, project.id))
		.all();

	const outcomes = db
		.select()
		.from(learningOutcomes)
		.where(eq(learningOutcomes.projectId, project.id))
		.all();

	const tags = db
		.select()
		.from(projectTags)
		.where(eq(projectTags.projectId, project.id))
		.all();

	return {
		domain,
		project,
		milestones: milestonesWithProgress,
		prerequisites,
		resources,
		languages,
		outcomes,
		tags
	};
};

export const actions: Actions = {
	toggleMilestone: async ({ request, locals }) => {
		if (!locals.user) error(401, 'Not authenticated');

		const formData = await request.formData();
		const milestoneId = Number(formData.get('milestoneId'));
		const currentStatus = formData.get('currentStatus') as string;

		if (!milestoneId) error(400, 'Missing milestone ID');

		const newStatus = currentStatus === 'completed' ? 'not_started' : 'completed';

		const existing = db
			.select()
			.from(userProgress)
			.where(
				and(
					eq(userProgress.userId, locals.user.id),
					eq(userProgress.milestoneId, milestoneId)
				)
			)
			.get();

		if (existing) {
			db.update(userProgress)
				.set({
					status: newStatus,
					completedAt: newStatus === 'completed' ? new Date().toISOString() : null,
					updatedAt: new Date().toISOString()
				})
				.where(eq(userProgress.id, existing.id))
				.run();
		} else {
			db.insert(userProgress)
				.values({
					userId: locals.user.id,
					milestoneId,
					status: newStatus,
					completedAt: newStatus === 'completed' ? new Date().toISOString() : null
				})
				.run();
		}

		return { success: true };
	}
};
