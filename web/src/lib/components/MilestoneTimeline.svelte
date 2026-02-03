<script lang="ts">
	import { enhance } from '$app/forms';
	import HintAccordion from './HintAccordion.svelte';
	import ArchitectureDoc from './ArchitectureDoc.svelte';
	import SubmissionForm from './SubmissionForm.svelte';

	let {
		milestones,
		hasApiKey = false,
		projectSlug = '',
		hasArchDoc = false
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
				fileName: string;
				fileSize: number;
				status: string;
				createdAt: string;
				review: string | null;
			}>;
		}>;
		hasApiKey: boolean;
		projectSlug: string;
		hasArchDoc: boolean;
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
	<ArchitectureDoc {projectSlug} hasDoc={hasArchDoc} />
	{#each milestones as milestone, idx}
		{@const isCompleted = milestone.status === 'completed'}
		{@const isOpen = openMilestones.has(milestone.id)}
		{@const criteria = parseJson(milestone.acceptanceCriteria)}
		{@const pitfalls = parseJson(milestone.commonPitfalls)}
		{@const concepts = parseJson(milestone.concepts)}
		{@const skills = parseJson(milestone.skills)}
		{@const deliverablesList = parseJson(milestone.deliverables)}
		{@const submissionCount = milestone.submissions?.length || 0}
		{@const hasReview = milestone.submissions?.some((s) => s.review)}

		<div class="milestone" class:completed={isCompleted}>
			{#if idx < milestones.length - 1}
				<div class="timeline-line" class:completed={isCompleted}></div>
			{/if}

			<form method="POST" action="?/toggleMilestone" use:enhance>
				<input type="hidden" name="milestoneId" value={milestone.id} />
				<input type="hidden" name="currentStatus" value={milestone.status} />
				<button type="submit" class="milestone-marker" class:completed={isCompleted}>
					{#if isCompleted}
						<svg width="12" height="12" viewBox="0 0 12 12" fill="none">
							<path d="M10 3L4.5 8.5L2 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
					{:else}
						{idx + 1}
					{/if}
				</button>
			</form>

			<div class="milestone-card" class:active={isOpen}>
				<button class="milestone-header" onclick={() => toggleOpen(milestone.id)}>
					<div class="milestone-header-content">
						<div class="milestone-title-row">
							<span class="milestone-title">{milestone.title}</span>
							{#if isCompleted}
								<span class="status-chip done">Done</span>
							{:else if submissionCount > 0}
								<span class="status-chip submitted">Submitted</span>
							{:else}
								<span class="status-chip todo">To Do</span>
							{/if}
						</div>
						{#if milestone.description}
							<div class="milestone-desc">{milestone.description}</div>
						{/if}
						{#if !isOpen}
							<div class="milestone-quick-info">
								{#if deliverablesList.length > 0}
									<span class="quick-tag">
										<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8.878.392a1.75 1.75 0 0 0-1.756 0l-5.25 3.045A1.75 1.75 0 0 0 1 4.951v6.098c0 .624.332 1.2.872 1.514l5.25 3.045a1.75 1.75 0 0 0 1.756 0l5.25-3.045c.54-.313.872-.89.872-1.514V4.951c0-.624-.332-1.2-.872-1.514L8.878.392ZM8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"/></svg>
										{deliverablesList.length} deliverable{deliverablesList.length > 1 ? 's' : ''}
									</span>
								{/if}
								{#if criteria.length > 0}
									<span class="quick-tag">
										<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M2.5 1.75v11.5c0 .138.112.25.25.25h3.17a.75.75 0 0 1 0 1.5H2.75A1.75 1.75 0 0 1 1 13.25V1.75C1 .784 1.784 0 2.75 0h8.5C12.216 0 13 .784 13 1.75v7.736a.75.75 0 0 1-1.5 0V1.75a.25.25 0 0 0-.25-.25h-8.5a.25.25 0 0 0-.25.25Zm10.276 8.22a.75.75 0 0 1 1.06 0l2.224 2.224a.75.75 0 1 1-1.06 1.06l-1.195-1.194v4.69a.75.75 0 0 1-1.5 0v-4.69l-1.195 1.194a.75.75 0 1 1-1.06-1.06l2.224-2.224Z"/></svg>
										{criteria.length} criteria
									</span>
								{/if}
								{#if submissionCount > 0}
									<span class="quick-tag submission-tag">
										<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M3.5 9.75A.75.75 0 0 1 4.25 9h6.5a.75.75 0 0 1 0 1.5h-6.5A.75.75 0 0 1 3.5 9.75Zm0-3.5A.75.75 0 0 1 4.25 5.5h6.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1-.75-.75ZM3.5 3A.75.75 0 0 1 4.25 2.25h6.5a.75.75 0 0 1 0 1.5h-6.5A.75.75 0 0 1 3.5 3Z"/></svg>
										{submissionCount} submission{submissionCount > 1 ? 's' : ''}
									</span>
								{/if}
							</div>
						{/if}
					</div>
					<span class="milestone-toggle" class:open={isOpen}>
						<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
							<path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
						</svg>
					</span>
				</button>

				{#if isOpen}
					<div class="milestone-details">
						<!-- WHAT TO DO section -->
						{#if deliverablesList.length > 0 || criteria.length > 0}
							<div class="detail-group">
								<div class="group-header">
									<div class="group-icon deliverables-icon">
										<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8.878.392a1.75 1.75 0 0 0-1.756 0l-5.25 3.045A1.75 1.75 0 0 0 1 4.951v6.098c0 .624.332 1.2.872 1.514l5.25 3.045a1.75 1.75 0 0 0 1.756 0l5.25-3.045c.54-.313.872-.89.872-1.514V4.951c0-.624-.332-1.2-.872-1.514L8.878.392ZM8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"/></svg>
									</div>
									<span class="group-title">What to Build</span>
								</div>

								{#if deliverablesList.length > 0}
									<div class="checklist">
										{#each deliverablesList as item}
											<div class="checklist-item">
												<span class="check-icon deliverable-check">
													<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16A8 8 0 1 1 8 0a8 8 0 0 1 0 16Zm3.78-9.72a.751.751 0 0 0-1.06-1.06L7.25 8.689 5.28 6.72a.751.751 0 0 0-1.06 1.06l2.5 2.5a.75.75 0 0 0 1.06 0l4-4Z"/></svg>
												</span>
												<span>{item}</span>
											</div>
										{/each}
									</div>
								{/if}

								{#if criteria.length > 0}
									<div class="sub-section">
										<div class="sub-section-label">Acceptance Criteria</div>
										<div class="criteria-list">
											{#each criteria as item}
												<div class="criteria-item">
													<span class="criteria-bullet">
														<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"/></svg>
													</span>
													<span>{item}</span>
												</div>
											{/each}
										</div>
									</div>
								{/if}
							</div>
						{/if}

						<!-- SKILLS & CONCEPTS -->
						{#if skills.length > 0 || concepts.length > 0}
							<div class="detail-group">
								<div class="group-header">
									<div class="group-icon skills-icon">
										<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M7.775 3.275a.75.75 0 0 0 1.06 1.06l1.25-1.25a2 2 0 1 1 2.83 2.83l-1.25 1.25a.75.75 0 0 0 1.06 1.06l1.25-1.25a3.5 3.5 0 0 0-4.95-4.95l-1.25 1.25Zm-4.69 9.64a2 2 0 0 1 0-2.83l1.25-1.25a.75.75 0 0 0-1.06-1.06l-1.25 1.25a3.5 3.5 0 0 0 4.95 4.95l1.25-1.25a.75.75 0 0 0-1.06-1.06l-1.25 1.25a2 2 0 0 1-2.83 0ZM5.47 5.47a.75.75 0 0 1 1.06 0l4 4a.75.75 0 0 1-1.06 1.06l-4-4a.75.75 0 0 1 0-1.06Z"/></svg>
									</div>
									<span class="group-title">Skills & Concepts</span>
								</div>
								<div class="tags-container">
									{#each skills as skill}
										<span class="skill-tag">{skill}</span>
									{/each}
									{#each concepts as concept}
										<span class="concept-tag">{concept}</span>
									{/each}
								</div>
							</div>
						{/if}

						<!-- HELP section -->
						{#if milestone.hintsLevel1 || pitfalls.length > 0}
							<div class="detail-group">
								<div class="group-header">
									<div class="group-icon help-icon">
										<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.92 6.085h.001a.749.749 0 1 1-1.342-.67c.169-.339.436-.701.849-.977C6.845 4.16 7.369 4 8 4a2.756 2.756 0 0 1 1.637.525c.503.377.863.965.863 1.725 0 .448-.115.83-.329 1.15-.205.307-.478.513-.708.662-.04.027-.08.052-.118.076-.428.27-.525.39-.525.61a.75.75 0 0 1-1.5 0c0-.76.478-1.204.964-1.51.162-.103.283-.184.378-.252.118-.084.182-.146.222-.207a.72.72 0 0 0 .116-.378c0-.274-.12-.452-.294-.582A1.264 1.264 0 0 0 8 5.5c-.39 0-.622.1-.784.206a1.174 1.174 0 0 0-.376.39l-.04.075Zm.94 5.165a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5Z"/></svg>
									</div>
									<span class="group-title">Guidance</span>
								</div>

								{#if milestone.hintsLevel1}
									<HintAccordion
										level1={milestone.hintsLevel1}
									/>
								{/if}

								{#if pitfalls.length > 0}
									<div class="pitfalls-section">
										<div class="sub-section-label pitfall-label">
											<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"/></svg>
											Common Pitfalls
										</div>
										{#each pitfalls as item}
											<div class="pitfall-item">
												<span class="pitfall-bullet"></span>
												<span>{item}</span>
											</div>
										{/each}
									</div>
								{/if}
							</div>
						{/if}

						<!-- SUBMIT WORK section -->
						<div class="detail-group submit-group">
							<div class="group-header">
								<div class="group-icon submit-icon">
									<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14ZM11.78 4.72a.749.749 0 1 1-1.06 1.06L8.75 3.811V9.5a.75.75 0 0 1-1.5 0V3.811L5.28 5.78a.749.749 0 1 1-1.06-1.06l3.25-3.25a.749.749 0 0 1 1.06 0l3.25 3.25Z"/></svg>
								</div>
								<span class="group-title">Submit Your Work</span>
								{#if submissionCount > 0}
									<span class="submission-count">{submissionCount}</span>
								{/if}
							</div>
							<SubmissionForm
								milestoneId={milestone.id}
								submissions={milestone.submissions || []}
								{hasApiKey}
							/>
						</div>
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
		padding-left: 44px;
		padding-bottom: 28px;
	}

	.milestone:last-child {
		padding-bottom: 0;
	}

	.timeline-line {
		position: absolute;
		left: 13px;
		top: 32px;
		bottom: 0;
		width: 2px;
		background: var(--border);
		transition: background 0.3s ease;
	}

	.timeline-line.completed {
		background: linear-gradient(180deg, var(--accent-bright), var(--accent));
	}

	.milestone-marker {
		position: absolute;
		left: 0;
		top: 0;
		width: 28px;
		height: 28px;
		background: var(--bg-card);
		border: 2px solid var(--border);
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 12px;
		font-weight: 700;
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.2s ease;
		padding: 0;
		font-family: inherit;
		box-shadow: var(--shadow-sm);
	}

	.milestone-marker:hover {
		border-color: var(--accent-bright);
		color: var(--accent-bright);
		transform: scale(1.1);
	}

	.milestone-marker.completed {
		background: var(--accent-bright);
		border-color: var(--accent-bright);
		color: white;
		box-shadow: 0 0 0 3px rgba(63, 185, 80, 0.2);
	}

	.milestone-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
		transition: border-color 0.2s ease, box-shadow 0.2s ease;
	}

	.milestone-card.active {
		border-color: var(--text-muted);
		box-shadow: var(--shadow-md);
	}

	.milestone-header {
		padding: 16px 18px;
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

	.milestone-header-content {
		flex: 1;
		min-width: 0;
	}

	.milestone-title-row {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 4px;
	}

	.milestone-title {
		font-size: 14px;
		font-weight: 600;
	}

	.status-chip {
		padding: 2px 8px;
		border-radius: 10px;
		font-size: 11px;
		font-weight: 500;
		flex-shrink: 0;
	}

	.status-chip.todo {
		background: rgba(110, 118, 129, 0.15);
		color: var(--text-muted);
	}

	.status-chip.submitted {
		background: rgba(88, 166, 255, 0.15);
		color: var(--blue);
	}

	.status-chip.done {
		background: rgba(63, 185, 80, 0.15);
		color: var(--accent-bright);
	}

	.milestone-desc {
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.5;
	}

	.milestone-quick-info {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-top: 8px;
		flex-wrap: wrap;
	}

	.quick-tag {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		color: var(--text-muted);
	}

	.quick-tag.submission-tag {
		color: var(--blue);
	}

	.milestone-toggle {
		color: var(--text-muted);
		flex-shrink: 0;
		margin-top: 4px;
		transition: transform 0.2s ease;
		display: flex;
	}

	.milestone-toggle.open {
		transform: rotate(180deg);
	}

	.milestone-details {
		border-top: 1px solid var(--border);
		padding: 0;
	}

	/* Detail Groups */
	.detail-group {
		padding: 18px 20px;
		border-bottom: 1px solid var(--border);
	}

	.detail-group:last-child {
		border-bottom: none;
	}

	.group-header {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 14px;
	}

	.group-icon {
		width: 28px;
		height: 28px;
		border-radius: 6px;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
	}

	.deliverables-icon {
		background: rgba(63, 185, 80, 0.12);
		color: var(--accent-bright);
	}

	.skills-icon {
		background: rgba(88, 166, 255, 0.12);
		color: var(--blue);
	}

	.help-icon {
		background: rgba(210, 153, 34, 0.12);
		color: var(--orange);
	}

	.submit-icon {
		background: rgba(163, 113, 247, 0.12);
		color: var(--purple);
	}

	.group-title {
		font-size: 13px;
		font-weight: 600;
		color: var(--text-primary);
		letter-spacing: 0.01em;
	}

	.submission-count {
		background: var(--purple);
		color: white;
		font-size: 10px;
		font-weight: 600;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	/* Checklist */
	.checklist {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.checklist-item {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		padding: 8px 10px;
		border-radius: 6px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.5;
		transition: background 0.15s;
	}

	.checklist-item:hover {
		background: var(--bg-dark);
	}

	.check-icon {
		flex-shrink: 0;
		margin-top: 1px;
	}

	.deliverable-check {
		color: var(--accent-bright);
		opacity: 0.6;
	}

	/* Sub-sections */
	.sub-section {
		margin-top: 14px;
		padding-top: 14px;
		border-top: 1px dashed var(--border);
	}

	.sub-section-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 8px;
	}

	.criteria-list {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.criteria-item {
		display: flex;
		align-items: flex-start;
		gap: 8px;
		padding: 6px 10px;
		border-radius: 6px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.5;
	}

	.criteria-bullet {
		color: var(--accent-bright);
		flex-shrink: 0;
		margin-top: 2px;
	}

	/* Tags */
	.tags-container {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.skill-tag {
		padding: 4px 12px;
		background: rgba(88, 166, 255, 0.08);
		border: 1px solid rgba(88, 166, 255, 0.2);
		border-radius: 20px;
		font-size: 12px;
		color: var(--blue);
	}

	.concept-tag {
		padding: 4px 12px;
		background: rgba(163, 113, 247, 0.08);
		border: 1px solid rgba(163, 113, 247, 0.2);
		border-radius: 20px;
		font-size: 12px;
		color: var(--purple);
	}

	/* Pitfalls */
	.pitfalls-section {
		margin-top: 14px;
		padding: 12px;
		background: rgba(210, 153, 34, 0.05);
		border: 1px solid rgba(210, 153, 34, 0.15);
		border-radius: var(--radius-md);
	}

	.pitfall-label {
		color: var(--orange);
		display: flex;
		align-items: center;
		gap: 6px;
		margin-bottom: 10px;
	}

	.pitfall-item {
		display: flex;
		align-items: flex-start;
		gap: 8px;
		padding: 5px 10px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.5;
	}

	.pitfall-bullet {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--orange);
		flex-shrink: 0;
		margin-top: 7px;
	}

	/* Submit group */
	.submit-group {
		background: rgba(163, 113, 247, 0.03);
	}

	@media (max-width: 768px) {
		.milestone {
			padding-left: 36px;
		}

		.timeline-line {
			left: 9px;
		}

		.milestone-marker {
			width: 22px;
			height: 22px;
			font-size: 10px;
		}

		.detail-group {
			padding: 14px 16px;
		}

		.milestone-title-row {
			flex-wrap: wrap;
		}
	}
</style>
