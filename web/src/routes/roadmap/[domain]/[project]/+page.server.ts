import { error, fail } from '@sveltejs/kit';
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
	projectTags,
	submissions,
	userSettings,
	aiInteractions
} from '$lib/server/db/schema.js';
import { eq, and, desc } from 'drizzle-orm';

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

	// Get submissions for each milestone
	const userSubmissions = locals.user
		? db
				.select()
				.from(submissions)
				.where(eq(submissions.userId, locals.user.id))
				.orderBy(desc(submissions.createdAt))
				.all()
		: [];

	const submissionsMap = new Map<number, typeof userSubmissions>();
	for (const sub of userSubmissions) {
		const existing = submissionsMap.get(sub.milestoneId) || [];
		existing.push(sub);
		submissionsMap.set(sub.milestoneId, existing);
	}

	// Get AI reviews for submissions
	const allSubIds = userSubmissions.map((s) => s.id);
	const reviews =
		allSubIds.length > 0 && locals.user
			? db
					.select()
					.from(aiInteractions)
					.where(
						and(
							eq(aiInteractions.userId, locals.user.id),
							eq(aiInteractions.interactionType, 'review')
						)
					)
					.orderBy(desc(aiInteractions.createdAt))
					.all()
			: [];

	const reviewsMap = new Map<number, (typeof reviews)[0]>();
	for (const r of reviews) {
		if (r.submissionId && !reviewsMap.has(r.submissionId)) {
			reviewsMap.set(r.submissionId, r);
		}
	}

	// Check if user has API key configured
	const hasApiKey = locals.user
		? !!(db
				.select()
				.from(userSettings)
				.where(eq(userSettings.userId, locals.user.id))
				.get()?.geminiApiKey)
		: false;

	const milestonesData = milestonesWithProgress.map((m) => {
		const milestoneSubmissions = (submissionsMap.get(m.id) || []).map((s) => ({
			...s,
			review: reviewsMap.get(s.id)?.response || null
		}));
		return {
			...m,
			submissions: milestoneSubmissions
		};
	});

	return {
		domain,
		project,
		milestones: milestonesData,
		prerequisites,
		resources,
		languages,
		outcomes,
		tags,
		hasApiKey
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
	},

	submitWork: async ({ request, locals }) => {
		if (!locals.user) error(401, 'Not authenticated');

		const formData = await request.formData();
		const milestoneId = Number(formData.get('milestoneId'));
		const content = (formData.get('content') as string)?.trim();
		const language = (formData.get('language') as string)?.trim() || null;

		if (!milestoneId) return fail(400, { submitError: 'Missing milestone ID' });
		if (!content) return fail(400, { submitError: 'Submission content is required' });

		const result = db
			.insert(submissions)
			.values({
				userId: locals.user.id,
				milestoneId,
				content,
				language
			})
			.returning()
			.get();

		return { submitSuccess: true, submissionId: result.id };
	}
};
