export interface User {
	id: number;
	username: string;
	email: string;
}

export interface Domain {
	id: number;
	slug: string;
	name: string;
	description: string | null;
	icon: string | null;
	sortOrder: number;
	projectCount?: number;
	completedMilestones?: number;
	totalMilestones?: number;
}

export interface Project {
	id: number;
	domainId: number;
	slug: string;
	name: string;
	description: string | null;
	difficulty: string;
	sortOrder: number;
	estimatedHours: string | null;
	essence: string | null;
	whyImportant: string | null;
	architectureDocPath: string | null;
	completedMilestones?: number;
	totalMilestones?: number;
	prerequisites?: { name: string; type: string | null }[];
	resources?: { title: string; url: string; type: string | null }[];
	languages?: { language: string; recommended: boolean }[];
	learningOutcomes?: string[];
	tags?: string[];
}

export interface Milestone {
	id: number;
	projectId: number;
	title: string;
	description: string | null;
	sortOrder: number;
	acceptanceCriteria: string | null;
	commonPitfalls: string | null;
	hintsLevel1: string | null;
	hintsLevel2: string | null;
	hintsLevel3: string | null;
	concepts: string | null;
	skills: string | null;
	deliverables: string | null;
	status?: string;
}

export interface ProgressSummary {
	domainSlug: string;
	completed: number;
	total: number;
}

export interface FileTreeNode {
	name: string;
	path: string;
	type: 'file' | 'directory';
	children?: FileTreeNode[];
	size?: number;
	extension?: string;
}
