import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import { migrate } from 'drizzle-orm/better-sqlite3/migrator';
import * as schema from './schema.js';
import { yamlProjectSchema, type YamlProject } from './yaml-schema.js';
import { readFileSync, readdirSync } from 'fs';
import { resolve, basename } from 'path';
import yaml from 'js-yaml';

const dbPath = resolve('data/editutor.db');
const projectsDataDir = resolve('../data/projects_data');

console.log('Seeding database...');
console.log('DB path:', dbPath);
console.log('Projects data dir:', projectsDataDir);

// Create DB connection
const sqlite = new Database(dbPath);
sqlite.pragma('journal_mode = WAL');
sqlite.pragma('foreign_keys = ON');

const db = drizzle(sqlite, { schema });

// Run migrations
migrate(db, { migrationsFolder: resolve('drizzle') });
console.log('Migrations applied.');

// Load all project YAML files
const projectFiles = readdirSync(projectsDataDir).filter((f) => f.endsWith('.yaml'));
console.log(`Found ${projectFiles.length} project files`);

// Parse and validate all projects
const projects: YamlProject[] = [];
const parseErrors: string[] = [];

for (const file of projectFiles) {
	const filePath = resolve(projectsDataDir, file);
	const content = readFileSync(filePath, 'utf-8');
	const rawData = yaml.load(content);

	const parseResult = yamlProjectSchema.safeParse(rawData);
	if (!parseResult.success) {
		parseErrors.push(
			`${file}: ${parseResult.error.issues.slice(0, 3).map((i) => i.message).join(', ')}`
		);
		continue;
	}

	projects.push(parseResult.data);
}

if (parseErrors.length > 0) {
	console.error('YAML validation errors:');
	for (const err of parseErrors.slice(0, 10)) {
		console.error(`  ${err}`);
	}
	console.error(`(${parseErrors.length} total errors)`);
	process.exit(1);
}

console.log(`Loaded ${projects.length} valid projects`);

// Group projects by domain
const domainMap = new Map<
	string,
	{ name: string; icon: string; subdomains: string[]; projects: Map<string, YamlProject[]> }
>();

// Domain metadata (hardcoded for now, could be moved to a separate file)
const domainMetadata: Record<
	string,
	{ name: string; icon: string; subdomains: string[] }
> = {
	'app-dev': {
		name: 'Application Development',
		icon: 'lucide:globe',
		subdomains: ['Web Apps', 'Mobile', 'Desktop', 'APIs']
	},
	systems: {
		name: 'Systems & Low-Level',
		icon: 'lucide:cpu',
		subdomains: ['Operating Systems', 'Networks', 'Compilers', 'Databases']
	},
	'data-storage': {
		name: 'Data & Storage',
		icon: 'lucide:database',
		subdomains: ['File Systems', 'Caching', 'Indexing', 'Compression']
	},
	distributed: {
		name: 'Distributed & Cloud',
		icon: 'lucide:network',
		subdomains: ['Consensus', 'Replication', 'Sharding', 'Messaging']
	},
	'ai-ml': {
		name: 'AI & Machine Learning',
		icon: 'lucide:brain',
		subdomains: ['ML Fundamentals', 'Deep Learning', 'NLP', 'Computer Vision']
	},
	'game-dev': {
		name: 'Game Development',
		icon: 'lucide:gamepad-2',
		subdomains: ['Game Engines', 'Multiplayer', 'Physics', 'Rendering']
	},
	compilers: {
		name: 'Languages & Compilers',
		icon: 'lucide:code-2',
		subdomains: ['Lexers', 'Parsers', 'Interpreters', 'Compilers']
	},
	security: {
		name: 'Security',
		icon: 'lucide:shield',
		subdomains: ['Cryptography', 'Authentication', 'Vulnerability Analysis']
	},
	specialized: {
		name: 'Specialized',
		icon: 'lucide:sparkles',
		subdomains: ['Emulators', 'Browsers', 'Advanced Systems']
	},
	'software-engineering': {
		name: 'Software Engineering Practices',
		icon: 'lucide:git-branch',
		subdomains: ['Testing', 'CI/CD', 'Observability', 'Architecture']
	},
	'world-scale': {
		name: 'World-Scale Infrastructure (Final Boss)',
		icon: 'lucide:trophy',
		subdomains: ['Massive Scale', 'Global Distribution', 'Enterprise']
	},
	'performance-engineering': {
		name: 'Performance Engineering',
		icon: 'lucide:gauge',
		subdomains: ['Low-Latency', 'Memory Optimization', 'Profiling', 'SIMD/Vectorization']
	}
};

// Group projects by domain and level
for (const proj of projects) {
	const domainId = proj.domain || 'tools';
	const level = proj.difficulty || 'intermediate';

	if (!domainMap.has(domainId)) {
		const meta = domainMetadata[domainId] || {
			name: domainId,
			icon: 'lucide:folder',
			subdomains: []
		};
		domainMap.set(domainId, {
			...meta,
			projects: new Map()
		});
	}

	const domain = domainMap.get(domainId)!;
	if (!domain.projects.has(level)) {
		domain.projects.set(level, []);
	}
	domain.projects.get(level)!.push(proj);
}

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
	for (const [domainId, domain] of domainMap) {
		const domainRow = db
			.insert(schema.domains)
			.values({
				slug: domainId,
				name: domain.name,
				description: domain.subdomains.join(', '),
				icon: domain.icon,
				sortOrder: domainOrder++
			})
			.returning()
			.get();

		let projectOrder = 0;
		const levels = ['beginner', 'intermediate', 'advanced', 'expert'] as const;

		for (const level of levels) {
			const projectsList = domain.projects.get(level) || [];
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
						estimatedHours: proj.estimated_hours ? String(proj.estimated_hours) : null,
						essence: proj.essence || null,
						whyImportant: proj.why_important || null,
						bridge: 0,
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
								: 'name' in prereq
									? prereq.name
									: (prereq.id ?? String(prereq));
						const type = typeof prereq === 'string' ? null : prereq?.type || null;
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
						const outcomeStr =
							typeof outcome === 'string' ? outcome : JSON.stringify(outcome);
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
								estimatedHours: ms.estimated_hours ? String(ms.estimated_hours) : null,
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
