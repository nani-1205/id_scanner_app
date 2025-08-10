// mongo-init/init-mongo.js

db = db.getSiblingDB(process.env.MONGO_DB);

// Create the application user with read/write permissions
db.createUser({
  user: process.env.MONGO_USER,
  pwd: process.env.MONGO_PASS,
  roles: [{ role: 'readWrite', db: process.env.MONGO_DB }],
});

print(`--- Successfully created user '${process.env.MONGO_USER}' ---`);

// Create collections with indexes for faster querying
db.createCollection('processed_documents');
db.processed_documents.createIndex({ "job_id": 1 });
db.processed_documents.createIndex({ "user_id": 1 });
db.processed_documents.createIndex({ "document_type": 1 });
db.processed_documents.createIndex({ "$**": "text" }); // For text search

print('--- Successfully created collections and indexes ---');