// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'id-scanner', // A more descriptive name for your app
    script: 'app.py',   // The script to run
    interpreter: '/root/id_scanner_app/venv/bin/python', // ABSOLUTE path to your venv's python
    // Optional: set environment variables from your .env file
    // env: {
    //   NODE_ENV: 'development' // Example, not strictly needed for this app
    // },
  }]
};