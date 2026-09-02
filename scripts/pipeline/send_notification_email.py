#!/usr/bin/env python3

import argparse
import smtplib
import os
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ─────────────────────────────────────────────
# Configuration — edit these or use env variables
# ─────────────────────────────────────────────
SMTP_SERVER   = os.environ.get("SMTP_SERVER",   "srelay.in2p3.fr")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER     = os.environ.get("SMTP_USER",     "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SENDER_EMAIL  = os.environ.get("SENDER_EMAIL",  SMTP_USER)

if not SENDER_EMAIL:
    SENDER_EMAIL = "fleg@lpnhe.in2p3.fr"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a notification email with optional attachment."
    )
    parser.add_argument("-s", "--subject",    required=True,  help="Email subject")
    parser.add_argument("-r", "--recipient",  required=True,  help="Recipient email address")
    parser.add_argument("-b", "--body",       required=False, help="Email body text", default="")
    parser.add_argument("-a", "--attachment", required=False, help="Path to file attachment", default=None)
    return parser.parse_args()


def build_message(subject, recipient, body, attachment_path=None):
    msg = MIMEMultipart()
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = recipient
    msg["Subject"] = subject

    # ── Body ──────────────────────────────────
    msg.attach(MIMEText(body, "plain"))

    # ── Attachment ────────────────────────────
    if attachment_path:
        if not os.path.isfile(attachment_path):
            print(f"[ERROR] Attachment file not found: {attachment_path}", file=sys.stderr)
            sys.exit(1)

        with open(attachment_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(attachment_path)}"
        )
        msg.attach(part)

    return msg


def send_email2(msg, recipient):
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
        print(f"[OK] Email sent to {recipient}")

    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Authentication failed. Check SMTP_USER and SMTP_PASSWORD.", file=sys.stderr)
        sys.exit(1)
    except smtplib.SMTPException as e:
        print(f"[ERROR] SMTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] Network error: {e}", file=sys.stderr)
        sys.exit(1)

def send_email(msg, recipient):
    try:
        # ── Choose connection type based on port ───
        if SMTP_PORT == 465:
            # SSL from the start
            server_context = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            # Plain or STARTTLS (port 25 or 587)
            server_context = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)

        with server_context as server:
            # Upgrade to TLS if supported (port 587)
            if SMTP_PORT == 587:
                server.starttls()

            # ── Login only if credentials are provided ──
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            else:
                print("[INFO] No credentials provided, skipping authentication.")

            server.sendmail(SENDER_EMAIL, recipient, msg.as_string())

        print(f"[OK] Email sent to {recipient}")

    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Authentication failed. Check SMTP_USER and SMTP_PASSWORD.", file=sys.stderr)
        sys.exit(1)
    except smtplib.SMTPException as e:
        print(f"[ERROR] SMTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] Network error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()

    msg = build_message(
        subject         = args.subject,
        recipient       = args.recipient,
        body            = args.body,
        attachment_path = args.attachment
    )

    send_email(msg, args.recipient)


if __name__ == "__main__":
    main()
