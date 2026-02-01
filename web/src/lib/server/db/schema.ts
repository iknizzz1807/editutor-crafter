import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';

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

export const sessions = sqliteTable('sessions', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	userId: integer('user_id')
		.notNull()
		.references(() => users.id, { onDelete: 'cascade' }),
	token: text('token').notNull().unique(),
	expiresAt: text('expires_at').notNull(),
	createdAt: text('created_at')
		.notNull()
		.$defaultFn(() => new Date().toISOString())
});

export const domains = sqliteTable('domains', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	slug: text('slug').notNull().unique(),
	name: text('name').notNull(),
	description: text('description'),
	icon: text('icon'),
	sortOrder: integer('sort_order').notNull().default(0)
});

export const projects = sqliteTable('projects', {
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
	whyImportant: text('why_important')
});

export const milestones = sqliteTable('milestones', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	projectId: integer('project_id')
		.notNull()
		.references(() => projects.id, { onDelete: 'cascade' }),
	title: text('title').notNull(),
	description: text('description'),
	sortOrder: integer('sort_order').notNull().default(0),
	acceptanceCriteria: text('acceptance_criteria'),
	commonPitfalls: text('common_pitfalls'),
	hintsLevel1: text('hints_level1'),
	hintsLevel2: text('hints_level2'),
	hintsLevel3: text('hints_level3'),
	concepts: text('concepts'),
	skills: text('skills'),
	deliverables: text('deliverables')
});

export const projectPrerequisites = sqliteTable('project_prerequisites', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	projectId: integer('project_id')
		.notNull()
		.references(() => projects.id, { onDelete: 'cascade' }),
	prerequisiteName: text('prerequisite_name').notNull(),
	prerequisiteType: text('prerequisite_type')
});

export const projectResources = sqliteTable('project_resources', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	projectId: integer('project_id')
		.notNull()
		.references(() => projects.id, { onDelete: 'cascade' }),
	title: text('title').notNull(),
	url: text('url').notNull(),
	resourceType: text('resource_type')
});

export const projectLanguages = sqliteTable('project_languages', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	projectId: integer('project_id')
		.notNull()
		.references(() => projects.id, { onDelete: 'cascade' }),
	language: text('language').notNull(),
	recommended: integer('recommended').notNull().default(0)
});

export const learningOutcomes = sqliteTable('learning_outcomes', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	projectId: integer('project_id')
		.notNull()
		.references(() => projects.id, { onDelete: 'cascade' }),
	outcome: text('outcome').notNull()
});

export const projectTags = sqliteTable('project_tags', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	projectId: integer('project_id')
		.notNull()
		.references(() => projects.id, { onDelete: 'cascade' }),
	tag: text('tag').notNull()
});

export const userProgress = sqliteTable('user_progress', {
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
});

export const aiInteractions = sqliteTable('ai_interactions', {
	id: integer('id').primaryKey({ autoIncrement: true }),
	userId: integer('user_id')
		.notNull()
		.references(() => users.id, { onDelete: 'cascade' }),
	projectId: integer('project_id').references(() => projects.id),
	milestoneId: integer('milestone_id').references(() => milestones.id),
	interactionType: text('interaction_type').notNull(),
	prompt: text('prompt').notNull(),
	response: text('response'),
	createdAt: text('created_at')
		.notNull()
		.$defaultFn(() => new Date().toISOString())
});
