// This script runs on first-time initialization to create the application user

db = db.getSiblingDB(process.env.MONGO_DB);

db.createUser({
  user: process.env.MONGO_USER,
  pwd: process.env.MONGO_PASS,
  roles: [
    {
      role: 'readWrite',
      db: process.env.MONGO_DB,
    },
  ],
});

print(`--- Successfully created user '${process.env.MONGO_USER}' in database '${process.env.MONGO_DB}' ---`);