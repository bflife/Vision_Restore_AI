# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability within Vision-Restore AI, please follow these steps:

### 1. Do Not Open a Public Issue

Please **do not** report security vulnerabilities through public GitHub issues.

### 2. Report Privately

Send a detailed report to the maintainers by:

- Opening a [private security advisory](https://github.com/Rav-xyl/vision-restore-ai/security/advisories/new) on GitHub
- Or emailing the maintainers directly (if email is provided)

### 3. Include in Your Report

- Type of vulnerability
- Full path to the affected file(s)
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 4. Response Timeline

- We will acknowledge receipt within 48 hours
- We will provide a detailed response within 7 days
- We will work with you to understand and resolve the issue

## Security Best Practices for Users

### Model Downloads

- Vision-Restore AI downloads AI models from official sources only
- All model URLs are hardcoded to official GitHub releases
- Verify checksums when available

### Local Processing

- All image processing happens locally on your machine
- No data is sent to external servers
- No analytics or telemetry is collected

### File Access

- The application only accesses files you explicitly select
- Configuration is stored in `~/.vision-restore/`
- No data is written outside these locations

## Acknowledgments

We appreciate security researchers who help keep Vision-Restore AI safe. Contributors who report valid security issues will be acknowledged in our release notes (unless they prefer to remain anonymous).
