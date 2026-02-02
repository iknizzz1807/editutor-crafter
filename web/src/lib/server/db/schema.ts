import { sqliteTable, text, integer, uniqueIndex, index } from 'drizzle-orm/sqlite-core';

export const users = sqliteTable('users', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	username: text('username').notNull().unique(),
	email: text('email').notNull().unique(),
	passwordHash: text('password_hash').notNull(),
	createdAt: text('created_at')
		.notNull()
		.$defaultFn(() => new Date().toISOString()),
	updatedAt: text('updated_at')
		.notNull()
		.$defaultFn(() => new Date().toISOString())
});

export const sessions = sqliteTable(
	'sessions',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		userId: integer('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		token: text('token').notNull().unique(),
		expiresAt: text('expires_at').notNull(),
		createdAt: text('created_at')
			.notNull()
			.$defaultFn(() => new Date().toISOString())
	},
	(table) => [index('idx_sessions_user_id').on(table.userId)]
);

export const domains = sqliteTable('domains', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	slug: text('slug').notNull().unique(),
	name: text('name').notNull(),
	description: text('description'),
	icon: text('icon'),
	sortOrder: integer('sort_order').notNull().default(0)
});

export const projects = sqliteTable(
	'projects',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		domainId: integer('domain_id')
			.notNull()
			.references(() => domains.id, { onDelete: 'cascade' }),
		slug: text('slug').notNull().unique(),
		name: text('name').notNull(),
		description: text('description'),
		difficulty: text('difficulty').notNull(),
		sortOrder: integer('sort_order').notNull().default(0),
		estimatedHours: text('estimated_hours'),
		essence: text('essence'),
		whyImportant: text('why_important'),
		bridge: integer('bridge').notNull().default(0)
	},
	(table) => [index('idx_projects_domain_id').on(table.domainId)]
);

export const milestones = sqliteTable(
	'milestones',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		projectId: integer('project_id')
			.notNull()
			.references(() => projects.id, { onDelete: 'cascade' }),
		slug: text('slug'),
		title: text('title').notNull(),
		description: text('description'),
		sortOrder: integer('sort_order').notNull().default(0),
		estimatedHours: text('estimated_hours'),
		acceptanceCriteria: text('acceptance_criteria'),
		commonPitfalls: text('common_pitfalls'),
		hintsLevel1: text('hints_level1'),
		hintsLevel2: text('hints_level2'),
		hintsLevel3: text('hints_level3'),
		concepts: text('concepts'),
		skills: text('skills'),
		deliverables: text('deliverables')
	},
	(table) => [index('idx_milestones_project_id').on(table.projectId)]
);

export const projectPrerequisites = sqliteTable(
	'project_prerequisites',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		projectId: integer('project_id')
			.notNull()
			.references(() => projects.id, { onDelete: 'cascade' }),
		prerequisiteName: text('prerequisite_name').notNull(),
		prerequisiteType: text('prerequisite_type')
	},
	(table) => [index('idx_project_prerequisites_project_id').on(table.projectId)]
);

export const projectResources = sqliteTable(
	'project_resources',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		projectId: integer('project_id')
			.notNull()
			.references(() => projects.id, { onDelete: 'cascade' }),
		title: text('title').notNull(),
		url: text('url').notNull(),
		resourceType: text('resource_type')
	},
	(table) => [index('idx_project_resources_project_id').on(table.projectId)]
);

export const projectLanguages = sqliteTable(
	'project_languages',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		projectId: integer('project_id')
			.notNull()
			.references(() => projects.id, { onDelete: 'cascade' }),
		language: text('language').notNull(),
		recommended: integer('recommended').notNull().default(0)
	},
	(table) => [index('idx_project_languages_project_id').on(table.projectId)]
);

export const learningOutcomes = sqliteTable(
	'learning_outcomes',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		projectId: integer('project_id')
			.notNull()
			.references(() => projects.id, { onDelete: 'cascade' }),
		outcome: text('outcome').notNull()
	},
	(table) => [index('idx_learning_outcomes_project_id').on(table.projectId)]
);

export const projectTags = sqliteTable(
	'project_tags',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		projectId: integer('project_id')
			.notNull()
			.references(() => projects.id, { onDelete: 'cascade' }),
		tag: text('tag').notNull()
	},
	(table) => [index('idx_project_tags_project_id').on(table.projectId)]
);

export const userProgress = sqliteTable(
	'user_progress',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		userId: integer('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		milestoneId: integer('milestone_id')
			.notNull()
			.references(() => milestones.id, { onDelete: 'cascade' }),
		status: text('status').notNull().default('not_started'),
		completedAt: text('completed_at'),
		updatedAt: text('updated_at')
			.notNull()
			.$defaultFn(() => new Date().toISOString())
	},
	(table) => [
		index('idx_user_progress_user_id').on(table.userId),
		index('idx_user_progress_milestone_id').on(table.milestoneId),
		uniqueIndex('idx_user_progress_unique').on(table.userId, table.milestoneId)
	]
);

export const userSettings = sqliteTable('user_settings', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	userId: integer('user_id')
		.notNull()
		.references(() => users.id, { onDelete: 'cascade' })
		.unique(),
	geminiApiKey: text('gemini_api_key'),
	updatedAt: text('updated_at')
		.notNull()
		.$defaultFn(() => new Date().toISOString())
});

export const submissions = sqliteTable(
	'submissions',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		userId: integer('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		milestoneId: integer('milestone_id')
			.notNull()
			.references(() => milestones.id, { onDelete: 'cascade' }),
		content: text('content').notNull(),
		language: text('language'),
		status: text('status').notNull().default('pending'),
		createdAt: text('created_at')
			.notNull()
			.$defaultFn(() => new Date().toISOString())
	},
	(table) => [
		index('idx_submissions_user_id').on(table.userId),
		index('idx_submissions_milestone_id').on(table.milestoneId)
	]
);

export const aiInteractions = sqliteTable(
	'ai_interactions',
	{
		id: integer('id').primaryKey({ autoIncrement: true }),
		userId: integer('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		projectId: integer('project_id').references(() => projects.id),
		milestoneId: integer('milestone_id').references(() => milestones.id),
		submissionId: integer('submission_id').references(() => submissions.id),
		interactionType: text('interaction_type').notNull(),
		prompt: text('prompt').notNull(),
		response: text('response'),
		createdAt: text('created_at')
			.notNull()
			.$defaultFn(() => new Date().toISOString())
	},
	(table) => [index('idx_ai_interactions_user_id').on(table.userId)]
);
