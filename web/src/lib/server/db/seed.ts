import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import { migrate } from 'drizzle-orm/better-sqlite3/migrator';
import * as schema from './schema.js';
import { yamlDataSchema, type YamlData } from './yaml-schema.js';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import yaml from 'js-yaml';

const dbPath = resolve('data/editutor.db');
const yamlPath = resolve('../data/projects.yaml');

console.log('Seeding database...');
console.log('DB path:', dbPath);
console.log('YAML path:', yamlPath);

// Create DB connection
const sqlite = new Database(dbPath);
sqlite.pragma('journal_mode = WAL');
sqlite.pragma('foreign_keys = ON');

const db = drizzle(sqlite, { schema });

// Run migrations
migrate(db, { migrationsFolder: resolve('drizzle') });
console.log('Migrations applied.');

// Load and validate YAML
const yamlContent = readFileSync(yamlPath, 'utf-8');
const rawData = yaml.load(yamlContent);
const parseResult = yamlDataSchema.safeParse(rawData);

if (!parseResult.success) {
	console.error('YAML validation failed:');
	for (const issue of parseResult.error.issues.slice(0, 20)) {
		console.error(`  ${issue.path.join('.')}: ${issue.message}`);
	}
	console.error(`(${parseResult.error.issues.length} total issues)`);
	process.exit(1);
}

const data: YamlData = parseResult.data;
console.log(`Found ${data.domains.length} domains`);
console.log(`Found ${Object.keys(data.expert_projects || {}).length} expert projects`);

// Helper to normalize languages (handles both flat list and dict)
function normalizeLanguages(
	langs: { recommended?: string[]; also_possible?: string[] } | string[] | undefined
): { recommended: string[]; also_possible: string[] } {
	if (!langs) return { recommended: [], also_possible: [] };
	if (Array.isArray(langs)) {
		return { recommended: langs, also_possible: [] };
	}
	return {
		recommended: langs.recommended || [],
		also_possible: langs.also_possible || []
	};
}

// Seed inside a transaction — preserves user_progress and ai_interactions
const seedTransaction = sqlite.transaction(() => {
	// Delete content tables only — NOT user_progress or ai_interactions
	sqlite.exec('DELETE FROM project_tags');
	sqlite.exec('DELETE FROM learning_outcomes');
	sqlite.exec('DELETE FROM project_languages');
	sqlite.exec('DELETE FROM project_resources');
	sqlite.exec('DELETE FROM project_prerequisites');
	sqlite.exec('DELETE FROM milestones');
	sqlite.exec('DELETE FROM projects');
	sqlite.exec('DELETE FROM domains');

	let domainOrder = 0;
	for (const domain of data.domains) {
		const domainRow = db
			.insert(schema.domains)
			.values({
				slug: domain.id,
				name: domain.name,
				description: domain.subdomains
					.map((s) => (typeof s === 'string' ? s : s.name))
					.join(', '),
				icon: domain.icon,
				sortOrder: domainOrder++
			})
			.returning()
			.get();

		let projectOrder = 0;
		const levels = ['beginner', 'intermediate', 'advanced', 'expert'] as const;

		for (const level of levels) {
			const projectsList = domain.projects[level] || [];
			for (const proj of projectsList) {
				const projectRow = db
					.insert(schema.projects)
					.values({
						domainId: domainRow.id,
						slug: proj.id,
						name: proj.name,
						description: proj.description || null,
						difficulty: level,
						sortOrder: projectOrder++,
						estimatedHours: proj.estimated_hours || null,
						essence: proj.essence || null,
						whyImportant: proj.why_important || null,
						bridge: proj.difficulty === 'beginner' ? 0 : 0, // Simplified or use proj.bridge if added to schema
						architectureDocPath: proj.architecture_doc || null
					})
					.returning()
					.get();

				// Prerequisites
				if (proj.prerequisites) {
					for (const prereq of proj.prerequisites) {
						const name =
							typeof prereq === 'string'
								? prereq
								: ('name' in prereq ? prereq.name : prereq.id ?? String(prereq));
						const type = typeof prereq === 'string' ? null : (prereq?.type || null);
						if (!name) continue;
						db.insert(schema.projectPrerequisites)
							.values({
								projectId: projectRow.id,
								prerequisiteName: name,
								prerequisiteType: type
							})
							.run();
					}
				}

				// Resources
				if (proj.resources) {
					for (const res of proj.resources) {
						db.insert(schema.projectResources)
							.values({
								projectId: projectRow.id,
								title: res.name,
								url: res.url,
								resourceType: res.type || null
							})
							.run();
					}
				}

				// Languages
				const langs = normalizeLanguages(proj.languages);
				for (const lang of langs.recommended) {
					db.insert(schema.projectLanguages)
						.values({ projectId: projectRow.id, language: lang, recommended: 1 })
						.run();
				}
				for (const lang of langs.also_possible) {
					db.insert(schema.projectLanguages)
						.values({ projectId: projectRow.id, language: lang, recommended: 0 })
						.run();
				}

				// Learning outcomes
				if (proj.learning_outcomes) {
					for (const outcome of proj.learning_outcomes) {
						const outcomeStr = typeof outcome === 'string' ? outcome : JSON.stringify(outcome);
						db.insert(schema.learningOutcomes)
							.values({ projectId: projectRow.id, outcome: outcomeStr })
							.run();
					}
				}

				// Tags
				if (proj.tags) {
					for (const tag of proj.tags) {
						db.insert(schema.projectTags)
							.values({ projectId: projectRow.id, tag })
							.run();
					}
				}

				// Milestones
				if (proj.milestones) {
					let msOrder = 0;
					for (const ms of proj.milestones) {
						db.insert(schema.milestones)
							.values({
								projectId: projectRow.id,
								slug: ms.id || null,
								title: ms.name,
								description: ms.description || null,
								sortOrder: msOrder++,
								estimatedHours: ms.estimated_hours || null,
								acceptanceCriteria: ms.acceptance_criteria
									? JSON.stringify(ms.acceptance_criteria)
									: null,
								commonPitfalls: ms.pitfalls ? JSON.stringify(ms.pitfalls) : null,
								concepts: ms.concepts ? JSON.stringify(ms.concepts) : null,
								skills: ms.skills ? JSON.stringify(ms.skills) : null,
								deliverables: ms.deliverables ? JSON.stringify(ms.deliverables) : null
							})
							.run();
					}
				}
			}
		}

		console.log(`  Seeded domain: ${domain.name} (${projectOrder} projects)`);
	}
});

seedTransaction();

console.log('Seed complete!');
sqlite.close();
