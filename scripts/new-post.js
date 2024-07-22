import fs from "fs";
import path from "path";

function getDate() {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, "0");
  const day = String(today.getDate()).padStart(2, "0");

  return `${year}-${month}-${day}`;
}

const args = process.argv.slice(2);

if (args.length === 0) {
  console.error(`Error: No filename argument provided
Usage: npm run new-post -- <filename>`);
  process.exit(1); // Terminate the script and return error code 1
}

let fileName = args[0];

// Add .md extension if not present
const fileExtensionRegex = /\.(md|mdx)$/i;
if (!fileExtensionRegex.test(fileName)) {
  fileName += ".md";
}

// Remove file extension to get the folder name
const folderName = fileName.replace(fileExtensionRegex, "");

const targetDir = "./src/content/posts/";
const folderPath = path.join(targetDir, folderName);
const fullPath = path.join(folderPath, fileName);

if (fs.existsSync(fullPath)) {
  console.error(`Error: File ${fullPath} already exists`);
  process.exit(1);
}

// Create the folder if it doesn't exist
if (!fs.existsSync(folderPath)) {
  fs.mkdirSync(folderPath, { recursive: true });
}

const content = `---
title: ${args[0]}
published: ${getDate()}
description: ''
image: ''
tags: []
category: ''
draft: false 
---
`;

fs.writeFileSync(fullPath, content);

console.log(`Post ${fullPath} created`);
