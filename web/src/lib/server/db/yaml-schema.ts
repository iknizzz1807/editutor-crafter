import { z } from 'zod';

const yamlMilestoneSchema = z.object({
	id: z.string().optional(),
	name: z.string(),
	description: z.string().optional(),
	acceptance_criteria: z.array(z.string()).optional(),
	pitfalls: z.array(z.string()).optional(),
	hints: z
		.object({
			level1: z.string().optional(),
			level2: z.string().optional(),
			level3: z.string().optional()
		})
		.optional(),
	concepts: z.array(z.string()).optional(),
	skills: z.array(z.string()).optional(),
	deliverables: z.array(z.string()).optional(),
	estimated_hours: z.string().optional(),
	order: z.number().optional(),
	project_id: z.string().optional()
});

const yamlLanguagesSchema = z.union([
	z.object({
		recommended: z.array(z.string()).default([]),
		also_possible: z.array(z.string()).default([])
	}),
	z.array(z.string())
]);

const yamlExpertProjectSchema = z.object({
	id: z.string(),
	name: z.string(),
	description: z.string().optional(),
	difficulty: z.string().optional(),
	estimated_hours: z.string().optional(),
	essence: z.string().optional(),
	why_important: z.string().optional(),
	domain_id: z.string().optional(),
	difficulty_score: z.number().optional(),
	bridge: z.boolean().or(z.number()).optional(),
	prerequisites: z
		.array(
			z.union([
				z.string(),
				z.object({ name: z.string(), type: z.string().optional(), id: z.string().optional() }),
				z.object({ id: z.string(), type: z.string().optional() })
			])
		)
		.optional(),
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
	learning_outcomes: z.array(z.string()).optional(),
	skills: z.array(z.string()).optional(),
	tags: z.array(z.string()).optional(),
	milestones: z.array(yamlMilestoneSchema).optional()
});

const yamlDomainProjectStub = z.object({
	id: z.string(),
	name: z.string(),
	description: z.string().optional(),
	detailed: z.boolean().optional(),
	bridge: z.boolean().or(z.number()).optional(),
	languages: yamlLanguagesSchema.optional()
});

const yamlDomainSchema = z.object({
	id: z.string(),
	name: z.string(),
	icon: z.string(),
	subdomains: z.array(z.union([z.string(), z.object({ name: z.string() })])),
	projects: z.record(z.string(), z.array(yamlDomainProjectStub).default([]))
});

export const yamlDataSchema = z.object({
	domains: z.array(yamlDomainSchema),
	expert_projects: z.record(z.string(), yamlExpertProjectSchema)
});

export type YamlData = z.infer<typeof yamlDataSchema>;
export type YamlExpertProject = z.infer<typeof yamlExpertProjectSchema>;
export type YamlMilestone = z.infer<typeof yamlMilestoneSchema>;
export type YamlDomainProjectStub = z.infer<typeof yamlDomainProjectStub>;
