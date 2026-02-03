import { createHighlighterCore, type HighlighterCore } from 'shiki/core';
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript';

let highlighterPromise: Promise<HighlighterCore> | null = null;
let highlighterInstance: HighlighterCore | null = null;

const THEME = 'github-dark';

// Map language names to their shiki/langs/* dynamic imports
const LANG_IMPORTS: Record<string, () => Promise<unknown>> = {
	javascript: () => import('shiki/langs/javascript.mjs'),
	typescript: () => import('shiki/langs/typescript.mjs'),
	python: () => import('shiki/langs/python.mjs'),
	java: () => import('shiki/langs/java.mjs'),
	c: () => import('shiki/langs/c.mjs'),
	cpp: () => import('shiki/langs/cpp.mjs'),
	csharp: () => import('shiki/langs/csharp.mjs'),
	go: () => import('shiki/langs/go.mjs'),
	rust: () => import('shiki/langs/rust.mjs'),
	ruby: () => import('shiki/langs/ruby.mjs'),
	php: () => import('shiki/langs/php.mjs'),
	swift: () => import('shiki/langs/swift.mjs'),
	kotlin: () => import('shiki/langs/kotlin.mjs'),
	scala: () => import('shiki/langs/scala.mjs'),
	r: () => import('shiki/langs/r.mjs'),
	html: () => import('shiki/langs/html.mjs'),
	css: () => import('shiki/langs/css.mjs'),
	scss: () => import('shiki/langs/scss.mjs'),
	sass: () => import('shiki/langs/sass.mjs'),
	less: () => import('shiki/langs/less.mjs'),
	json: () => import('shiki/langs/json.mjs'),
	yaml: () => import('shiki/langs/yaml.mjs'),
	toml: () => import('shiki/langs/toml.mjs'),
	xml: () => import('shiki/langs/xml.mjs'),
	sql: () => import('shiki/langs/sql.mjs'),
	bash: () => import('shiki/langs/bash.mjs'),
	powershell: () => import('shiki/langs/powershell.mjs'),
	markdown: () => import('shiki/langs/markdown.mjs'),
	dockerfile: () => import('shiki/langs/dockerfile.mjs'),
	lua: () => import('shiki/langs/lua.mjs'),
	dart: () => import('shiki/langs/dart.mjs'),
	elixir: () => import('shiki/langs/elixir.mjs'),
	haskell: () => import('shiki/langs/haskell.mjs'),
	ocaml: () => import('shiki/langs/ocaml.mjs'),
	clojure: () => import('shiki/langs/clojure.mjs'),
	erlang: () => import('shiki/langs/erlang.mjs'),
	zig: () => import('shiki/langs/zig.mjs'),
	nim: () => import('shiki/langs/nim.mjs'),
	perl: () => import('shiki/langs/perl.mjs'),
	groovy: () => import('shiki/langs/groovy.mjs'),
	hcl: () => import('shiki/langs/hcl.mjs'),
	vue: () => import('shiki/langs/vue.mjs'),
	svelte: () => import('shiki/langs/svelte.mjs'),
	jsx: () => import('shiki/langs/jsx.mjs'),
	tsx: () => import('shiki/langs/tsx.mjs'),
};

function getHighlighter(): Promise<HighlighterCore> {
	if (highlighterInstance) return Promise.resolve(highlighterInstance);
	if (!highlighterPromise) {
		highlighterPromise = createHighlighterCore({
			engine: createJavaScriptRegexEngine(),
			themes: [import('shiki/themes/github-dark.mjs')],
			langs: []
		}).then((hl) => {
			highlighterInstance = hl;
			return hl;
		});
	}
	return highlighterPromise;
}

const loadedLangs = new Set<string>();

export async function highlightCode(code: string, lang?: string): Promise<string> {
	const hl = await getHighlighter();

	const language = lang?.toLowerCase() || 'text';

	if (language !== 'text' && language !== 'plaintext' && !loadedLangs.has(language)) {
		const langImport = LANG_IMPORTS[language];
		if (!langImport) {
			return escapeHtml(code);
		}
		try {
			await hl.loadLanguage(langImport() as never);
			loadedLangs.add(language);
		} catch {
			return escapeHtml(code);
		}
	}

	if (language === 'text' || language === 'plaintext' || !loadedLangs.has(language)) {
		return escapeHtml(code);
	}

	const html = hl.codeToHtml(code, {
		lang: language,
		theme: THEME
	});

	// Extract just the inner code content from shiki's output
	// Shiki wraps in <pre class="shiki ..."><code>...</code></pre>
	const match = html.match(/<code>([\s\S]*)<\/code>/);
	return match ? match[1] : escapeHtml(code);
}

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;');
}
