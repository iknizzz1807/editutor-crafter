<script lang="ts">
	import { enhance } from '$app/forms';
	import HintAccordion from './HintAccordion.svelte';
	import SubmissionForm from './SubmissionForm.svelte';

	let {
		milestones,
		languages = [],
		hasApiKey = false
	}: {
		milestones: Array<{
			id: number;
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
			status: string;
			submissions: Array<{
				id: number;
				content: string;
				language: string | null;
				status: string;
				createdAt: string;
				review: string | null;
			}>;
		}>;
		languages: Array<{ language: string; recommended: number }>;
		hasApiKey: boolean;
	} = $props();

	let openMilestones = $state(new Set<number>());

	function toggleOpen(id: number) {
		if (openMilestones.has(id)) {
			openMilestones.delete(id);
		} else {
			openMilestones.add(id);
		}
		openMilestones = new Set(openMilestones);
	}

	function parseJson(str: string | null): string[] {
		if (!str) return [];
		try {
			return JSON.parse(str);
		} catch {
			return [];
		}
	}
</script>

<div class="timeline">
	{#each milestones as milestone, idx}
		{@const isCompleted = milestone.status === 'completed'}
		{@const isOpen = openMilestones.has(milestone.id)}
		{@const criteria = parseJson(milestone.acceptanceCriteria)}
		{@const pitfalls = parseJson(milestone.commonPitfalls)}
		{@const concepts = parseJson(milestone.concepts)}
		{@const skills = parseJson(milestone.skills)}
		{@const deliverablesList = parseJson(milestone.deliverables)}

		<div class="milestone" class:completed={isCompleted}>
			{#if idx < milestones.length - 1}
				<div class="timeline-line" class:completed={isCompleted}></div>
			{/if}

			<form method="POST" action="?/toggleMilestone" use:enhance>
				<input type="hidden" name="milestoneId" value={milestone.id} />
				<input type="hidden" name="currentStatus" value={milestone.status} />
				<button type="submit" class="milestone-marker" class:completed={isCompleted}>
					{#if isCompleted}
						✓
					{:else}
						{idx + 1}
					{/if}
				</button>
			</form>

			<div class="milestone-card">
				<button class="milestone-header" onclick={() => toggleOpen(milestone.id)}>
					<div>
						<div class="milestone-title">{milestone.title}</div>
						{#if milestone.description}
							<div class="milestone-desc">{milestone.description}</div>
						{/if}
					</div>
					<span class="milestone-toggle" class:open={isOpen}>▼</span>
				</button>

				{#if isOpen}
					<div class="milestone-details">
						{#if skills.length > 0}
							<div class="detail-section">
								<div class="detail-section-title">🎯 Skills</div>
								<div class="skills-list">
									{#each skills as skill}
										<span class="skill-tag">{skill}</span>
									{/each}
								</div>
							</div>
						{/if}

						{#if deliverablesList.length > 0}
							<div class="detail-section">
								<div class="detail-section-title">📦 Deliverables</div>
								<ul class="detail-list">
									{#each deliverablesList as item}
										<li class="deliverable">{item}</li>
									{/each}
								</ul>
							</div>
						{/if}

						{#if criteria.length > 0}
							<div class="detail-section">
								<div class="detail-section-title">✅ Acceptance Criteria</div>
								<ul class="detail-list criteria">
									{#each criteria as item}
										<li>{item}</li>
									{/each}
								</ul>
							</div>
						{/if}

						{#if milestone.hintsLevel1 || milestone.hintsLevel2 || milestone.hintsLevel3}
							<div class="detail-section">
								<div class="detail-section-title">💡 Hints (Progressive)</div>
								<HintAccordion
									level1={milestone.hintsLevel1}
									level2={milestone.hintsLevel2}
									level3={milestone.hintsLevel3}
								/>
							</div>
						{/if}

						{#if pitfalls.length > 0}
							<div class="detail-section">
								<div class="detail-section-title">⚠️ Common Pitfalls</div>
								<ul class="detail-list pitfalls">
									{#each pitfalls as item}
										<li>{item}</li>
									{/each}
								</ul>
							</div>
						{/if}

						{#if concepts.length > 0}
							<div class="detail-section">
								<div class="detail-section-title">📚 Key Concepts</div>
								<div class="skills-list">
									{#each concepts as concept}
										<span class="skill-tag">{concept}</span>
									{/each}
								</div>
							</div>
						{/if}

						<SubmissionForm
							milestoneId={milestone.id}
							{languages}
							submissions={milestone.submissions || []}
							{hasApiKey}
						/>
					</div>
				{/if}
			</div>
		</div>
	{/each}
</div>

<style>
	.timeline {
		position: relative;
	}

	.milestone {
		position: relative;
		padding-left: 40px;
		padding-bottom: 24px;
	}

	.milestone:last-child {
		padding-bottom: 0;
	}

	.timeline-line {
		position: absolute;
		left: 11px;
		top: 28px;
		bottom: 0;
		width: 2px;
		background: var(--border);
	}

	.timeline-line.completed {
		background: var(--accent-bright);
	}

	.milestone-marker {
		position: absolute;
		left: 0;
		top: 0;
		width: 24px;
		height: 24px;
		background: var(--bg-card);
		border: 2px solid var(--border);
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 11px;
		font-weight: 700;
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.2s ease;
		padding: 0;
		font-family: inherit;
	}

	.milestone-marker:hover {
		border-color: var(--accent-bright);
		color: var(--accent-bright);
	}

	.milestone-marker.completed {
		background: var(--accent-bright);
		border-color: var(--accent-bright);
		color: white;
	}

	.milestone-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 8px;
		overflow: hidden;
	}

	.milestone-header {
		padding: 14px 16px;
		cursor: pointer;
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 12px;
		width: 100%;
		background: none;
		border: none;
		color: var(--text-primary);
		font-family: inherit;
		text-align: left;
	}

	.milestone-header:hover {
		background: var(--bg-card-hover);
	}

	.milestone-title {
		font-size: 14px;
		font-weight: 600;
		margin-bottom: 4px;
	}

	.milestone-desc {
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.5;
	}

	.milestone-toggle {
		color: var(--text-muted);
		font-size: 12px;
		flex-shrink: 0;
		margin-top: 2px;
		transition: transform 0.2s ease;
	}

	.milestone-toggle.open {
		transform: rotate(180deg);
	}

	.milestone-details {
		border-top: 1px solid var(--border);
		padding: 16px;
	}

	.detail-section {
		margin-bottom: 20px;
	}

	.detail-section:last-child {
		margin-bottom: 0;
	}

	.detail-section-title {
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 10px;
	}

	.skills-list {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.skill-tag {
		padding: 4px 10px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 4px;
		font-size: 12px;
		color: var(--text-secondary);
	}

	.detail-list {
		list-style: none;
	}

	.detail-list li {
		position: relative;
		padding: 8px 0 8px 24px;
		font-size: 13px;
		color: var(--text-secondary);
		border-bottom: 1px solid var(--border);
	}

	.detail-list li:last-child {
		border-bottom: none;
	}

	.detail-list li::before {
		content: '○';
		position: absolute;
		left: 0;
		color: var(--accent-bright);
	}

	.detail-list.criteria li::before {
		content: '✓';
		color: var(--beginner);
	}

	.detail-list.pitfalls li::before {
		content: '⚠';
		color: var(--orange);
	}

	@media (max-width: 768px) {
		.milestone {
			padding-left: 32px;
		}

		.timeline-line {
			left: 7px;
		}

		.milestone-marker {
			width: 18px;
			height: 18px;
			font-size: 10px;
		}
	}
</style>
