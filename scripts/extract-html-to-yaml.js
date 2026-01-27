#!/usr/bin/env node
/**
 * Extract embedded projectsData from visualizer.html and output as JSON
 * This JSON will then be merged into projects.yaml
 */

const fs = require('fs');
const path = require('path');

const htmlPath = path.join(__dirname, '..', 'visualizer.html');
const outputPath = path.join(__dirname, '..', 'data', 'extracted-projects.json');

// Read HTML file
const html = fs.readFileSync(htmlPath, 'utf-8');

// Find the projectsData object
// It starts with "const projectsData = {" and ends before "// State"
const startMarker = 'const projectsData = {';
const endMarker = '// State';

const startIdx = html.indexOf(startMarker);
const endIdx = html.indexOf(endMarker);

if (startIdx === -1 || endIdx === -1) {
    console.error('Could not find projectsData in HTML');
    process.exit(1);
}

// Extract just the object part (without "const projectsData = ")
const objectStart = startIdx + 'const projectsData = '.length;
let jsObjectStr = html.substring(objectStart, endIdx).trim();

// Remove trailing semicolon and closing brace issues
jsObjectStr = jsObjectStr.replace(/\s*;?\s*$/, '');
// Ensure it ends with }
if (!jsObjectStr.endsWith('}')) {
    // Find last }
    const lastBrace = jsObjectStr.lastIndexOf('}');
    if (lastBrace !== -1) {
        jsObjectStr = jsObjectStr.substring(0, lastBrace + 1);
    }
}

// Evaluate the JS object (it uses unquoted keys which aren't valid JSON)
// We'll use eval in a sandboxed way
try {
    // Use Function constructor to evaluate
    const projectsData = eval('(' + jsObjectStr + ')');

    // Output as JSON
    fs.writeFileSync(outputPath, JSON.stringify(projectsData, null, 2));
    console.log(`Extracted ${Object.keys(projectsData.expertProjects || {}).length} expert projects`);
    console.log(`Extracted ${(projectsData.domains || []).length} domains`);
    console.log(`Output written to: ${outputPath}`);
} catch (e) {
    console.error('Error parsing JS object:', e.message);
    // Write partial data for debugging
    fs.writeFileSync(outputPath + '.debug.txt', jsObjectStr.substring(0, 5000));
    process.exit(1);
}
