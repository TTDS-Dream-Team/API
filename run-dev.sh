poetry run uvicorn main:app --host 0.0.0.0 --port 4242 --reload --reload-dir /root/better_reads/API --ssl-keyfile=/etc/letsencrypt/live/api.better-reads.xyz/privkey.pem --ssl-certfile=/etc/letsencrypt/live/api.better-reads.xyz/cert.pem
