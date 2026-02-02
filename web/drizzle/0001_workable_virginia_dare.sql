ALTER TABLE `milestones` ADD `slug` text;--> statement-breakpoint
ALTER TABLE `milestones` ADD `estimated_hours` text;--> statement-breakpoint
CREATE INDEX `idx_milestones_project_id` ON `milestones` (`project_id`);--> statement-breakpoint
ALTER TABLE `projects` ADD `bridge` integer DEFAULT 0 NOT NULL;--> statement-breakpoint
CREATE INDEX `idx_projects_domain_id` ON `projects` (`domain_id`);--> statement-breakpoint
CREATE INDEX `idx_ai_interactions_user_id` ON `ai_interactions` (`user_id`);--> statement-breakpoint
CREATE INDEX `idx_learning_outcomes_project_id` ON `learning_outcomes` (`project_id`);--> statement-breakpoint
CREATE INDEX `idx_project_languages_project_id` ON `project_languages` (`project_id`);--> statement-breakpoint
CREATE INDEX `idx_project_prerequisites_project_id` ON `project_prerequisites` (`project_id`);--> statement-breakpoint
CREATE INDEX `idx_project_resources_project_id` ON `project_resources` (`project_id`);--> statement-breakpoint
CREATE INDEX `idx_project_tags_project_id` ON `project_tags` (`project_id`);--> statement-breakpoint
CREATE INDEX `idx_sessions_user_id` ON `sessions` (`user_id`);--> statement-breakpoint
CREATE INDEX `idx_user_progress_user_id` ON `user_progress` (`user_id`);--> statement-breakpoint
CREATE INDEX `idx_user_progress_milestone_id` ON `user_progress` (`milestone_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `idx_user_progress_unique` ON `user_progress` (`user_id`,`milestone_id`);