// mongo-init/init-mongo.js

// This script will create a new user and a new database

db = db.getSiblingDB(process.env.MONGO_DB); // Switch to or create the 'id_scanner_db'

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

print('Successfully created application user and database.');