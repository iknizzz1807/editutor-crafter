import { z } from 'zod';

const yamlLanguagesSchema = z.union([
	z.object({
		recommended: z.array(z.string()).default([]),
		also_possible: z.array(z.string()).default([])
	}),
	z.array(z.string())
]);

const yamlMilestoneSchema = z.object({
	id: z.string().optional(),
	name: z.string(),
	description: z.string().optional(),
	acceptance_criteria: z.array(z.string()).optional(),
	pitfalls: z.array(z.string()).optional(),
	concepts: z.array(z.string()).optional(),
	skills: z.array(z.string()).optional(),
	deliverables: z.array(z.string()).optional(),
	estimated_hours: z.union([z.string(), z.number()]).optional()
});

const yamlProjectSchema = z.object({
	id: z.string(),
	name: z.string(),
	description: z.string().optional(),
	difficulty: z.string().optional(),
	estimated_hours: z.union([z.string(), z.number()]).optional(),
	essence: z.string().optional(),
	why_important: z.string().optional(),
	learning_outcomes: z.array(z.union([z.string(), z.any()])).optional(),
	skills: z.array(z.string()).optional(),
	tags: z.array(z.string()).optional(),
	milestones: z.array(yamlMilestoneSchema).optional(),
	architecture_doc: z.string().optional(),
	languages: yamlLanguagesSchema.optional(),
	resources: z
		.array(
			z.object({
				name: z.string(),
				url: z.string(),
				type: z.string().optional()
			})
		)
		.optional(),
	prerequisites: z
		.array(
			z.union([
				z.string(),
				z.object({ name: z.string(), type: z.string().optional(), id: z.string().optional() }),
				z.object({ id: z.string(), type: z.string().optional() })
			])
		)
		.optional()
});

const yamlDomainSchema = z.object({
	id: z.string(),
	name: z.string(),
	icon: z.string(),
	subdomains: z.array(z.union([z.string(), z.object({ name: z.string() })])),
	projects: z.record(z.string(), z.array(yamlProjectSchema).default([]))
});

export const yamlDataSchema = z.object({
	domains: z.array(yamlDomainSchema)
});

export type YamlData = z.infer<typeof yamlDataSchema>;
export type YamlProject = z.infer<typeof yamlProjectSchema>;
export type YamlMilestone = z.infer<typeof yamlMilestoneSchema>;
