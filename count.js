const fs = require('fs');

// 1. Read the JSON file
const rawData = fs.readFileSync('bangla_output.json', 'utf8');

// 2. Parse the JSON data into a JavaScript array
const items = JSON.parse(rawData);

// 3. Extract all clipitem_ids and put them in a Set (which only keeps unique values)
const uniqueIds = new Set(items.map(item => item.clipitem_id));

// 4. Output the size of the Set
console.log(`Total number of different clipitem_ids: ${uniqueIds.size}`);
