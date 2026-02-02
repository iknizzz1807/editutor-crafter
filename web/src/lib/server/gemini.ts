import { GoogleGenerativeAI } from '@google/generative-ai';

export function createGeminiClient(apiKey: string) {
	return new GoogleGenerativeAI(apiKey);
}

interface MilestoneContext {
	title: string;
	description: string | null;
	acceptanceCriteria: string | null;
	concepts: string | null;
	skills: string | null;
	deliverables: string | null;
	commonPitfalls: string | null;
}

export async function reviewSubmission(
	apiKey: string,
	milestone: MilestoneContext,
	code: string,
	language: string | null
): Promise<string> {
	const client = createGeminiClient(apiKey);
	const model = client.getGenerativeModel({ model: 'gemini-2.0-flash' });

	const prompt = `You are an expert code reviewer and programming tutor. Review the following student submission for a learning milestone.

## Milestone: ${milestone.title}
${milestone.description ? `**Description:** ${milestone.description}` : ''}
${milestone.acceptanceCriteria ? `**Acceptance Criteria:** ${milestone.acceptanceCriteria}` : ''}
${milestone.concepts ? `**Key Concepts:** ${milestone.concepts}` : ''}
${milestone.skills ? `**Skills:** ${milestone.skills}` : ''}
${milestone.deliverables ? `**Deliverables:** ${milestone.deliverables}` : ''}
${milestone.commonPitfalls ? `**Common Pitfalls to Watch For:** ${milestone.commonPitfalls}` : ''}

## Student Submission${language ? ` (${language})` : ''}
\`\`\`${language || ''}
${code}
\`\`\`

Please provide a structured review with the following sections:
1. **Overall Assessment** - Brief summary of the submission quality and whether it meets the milestone requirements
2. **Strengths** - What the student did well
3. **Issues** - Problems, bugs, or areas that don't meet acceptance criteria
4. **Suggestions** - Specific, actionable improvements the student can make
5. **Score** - A score from 1-10 based on how well the submission meets the milestone requirements

Be constructive, encouraging, and educational in your feedback. Focus on helping the student learn.`;

	const result = await model.generateContent(prompt);
	return result.response.text();
}

export async function guideStudent(
	apiKey: string,
	milestone: MilestoneContext,
	question: string
): Promise<string> {
	const client = createGeminiClient(apiKey);
	const model = client.getGenerativeModel({ model: 'gemini-2.0-flash' });

	const prompt = `You are a patient and encouraging programming tutor. A student is working on a learning milestone and needs guidance.

## Milestone: ${milestone.title}
${milestone.description ? `**Description:** ${milestone.description}` : ''}
${milestone.acceptanceCriteria ? `**Acceptance Criteria:** ${milestone.acceptanceCriteria}` : ''}
${milestone.concepts ? `**Key Concepts:** ${milestone.concepts}` : ''}
${milestone.skills ? `**Skills:** ${milestone.skills}` : ''}
${milestone.deliverables ? `**Deliverables:** ${milestone.deliverables}` : ''}

## Student's Question
${question}

**Important guidelines:**
- Do NOT provide complete code solutions. Guide the student to figure it out themselves.
- Use the Socratic method: ask leading questions, provide hints, explain concepts.
- If they're stuck, break the problem into smaller steps.
- Reference relevant concepts and patterns they should explore.
- Encourage them to try and experiment.
- If they share code with bugs, point them in the right direction without fixing it for them.`;

	const result = await model.generateContent(prompt);
	return result.response.text();
}
