# JOSS Submission Checklist for EasyTrack

This document outlines the tasks required to submit EasyTrack to the Journal of Open Source Software (JOSS).

## Pre-submission Requirements

### Repository Requirements
- [x] Software has an **OSI-approved open source license** (LICENSE.md exists)
- [x] Software is stored in a **version-controlled repository** (GitHub)
- [ ] Software has a **clear statement of need** that describes the purpose
- [ ] Software is **feature-complete** and ready for use
- [ ] Software has **at least one release** with a DOI (e.g., via Zenodo)

### Documentation Requirements
- [ ] **Installation instructions** are complete and clear
- [ ] **Usage documentation** with examples is available
- [ ] **API documentation** (if applicable) is complete
- [ ] **Contribution guidelines** (CONTRIBUTING.md) are provided
- [ ] **Community guidelines** including:
  - [ ] How to report issues
  - [ ] How to contribute
  - [ ] How to seek support
- [ ] **Example usage** or tutorials are available

### Code Quality Requirements
- [ ] **Automated tests** are present and passing
- [ ] Test **coverage is documented**
- [ ] Code follows **best practices** for the language
- [ ] **Dependencies** are clearly specified
- [ ] Software is **platform-independent** or limitations are documented

### Paper Requirements (paper/paper.md)
- [x] Paper uses JOSS **markdown format**
- [x] Paper includes required **YAML header** with:
  - [x] Title
  - [x] Tags (at least 3)
  - [x] Authors with affiliations
  - [ ] Authors with valid **ORCIDs** (currently placeholder: 0000-0000-0000-0000)
  - [x] Date
  - [x] Bibliography reference
- [ ] **paper.bib** bibliography file exists
- [ ] Paper includes **Summary** section (250-1000 words)
- [ ] Paper includes **Statement of Need** section
- [ ] Paper describes **target audience**
- [ ] Paper explains **how it fits into research landscape**
- [ ] Paper includes **References** to relevant work
- [ ] Paper length is **between 250-1000 words** (excluding references)
- [ ] Paper includes **Acknowledgements** (if applicable)
- [ ] Paper includes **Figures** with proper captions (if needed)

### Author Information
- [ ] All authors have **valid ORCID IDs**
- [ ] Author **affiliations are complete** and accurate
- [ ] **Corresponding authors** are correctly designated
- [ ] All authors have **consented to submission**

## Submission Process

### Pre-submission Validation
- [ ] Run JOSS **paper validation locally**:
  ```bash
  docker run --rm \
    --volume $PWD/paper:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
  ```
- [ ] Ensure paper **compiles without errors**
- [ ] Check all **links in paper are valid**
- [ ] Verify **references are properly formatted**

### GitHub Repository Setup
- [ ] Create a **tagged release** (e.g., v1.0.0)
- [ ] Link release to **Zenodo** for DOI
- [ ] Add **JOSS badge** placeholder to README
- [ ] Ensure all **CI/CD tests pass**
- [ ] Repository has clear **README.md** with:
  - [ ] Project description
  - [ ] Installation instructions
  - [ ] Quick start guide
  - [ ] Link to full documentation
  - [ ] Citation information

### Code of Conduct
- [ ] Add **CODE_OF_CONDUCT.md** (can use Contributor Covenant)

### Contributing Guidelines
- [ ] Create **CONTRIBUTING.md** with:
  - [ ] How to set up development environment
  - [ ] How to run tests
  - [ ] Code style guidelines
  - [ ] Pull request process

### Citation Information
- [ ] Create **CITATION.cff** file for software citation
- [ ] Include citation information in README

## JOSS Submission

### Submit to JOSS
1. [ ] Go to https://joss.theoj.org/papers/new
2. [ ] Provide repository URL
3. [ ] Provide link to paper.md in repository
4. [ ] Fill out submission form
5. [ ] Submit paper

### Review Process
- [ ] Respond to **pre-review** comments
- [ ] Address **editor** feedback
- [ ] Respond to **reviewer comments**
- [ ] Make requested **code changes**
- [ ] Update **documentation** as needed
- [ ] Update **paper** based on feedback

### Post-Acceptance
- [ ] Add **JOSS DOI badge** to README
- [ ] Update **citation information**
- [ ] Announce publication on:
  - [ ] Project website
  - [ ] Social media
  - [ ] Relevant mailing lists

## Current Status Summary

### Completed
- Basic paper.md structure created
- Authors and affiliations added
- Tags defined
- License in place

### In Progress
- Paper content needs to be written
- ORCID IDs need to be verified/updated

### Not Started
- paper.bib bibliography file
- Documentation improvements
- Community guidelines (CONTRIBUTING.md, CODE_OF_CONDUCT.md)
- CITATION.cff file
- Zenodo integration for DOI
- JOSS paper content (Summary, Statement of Need, etc.)

## Resources

- **JOSS Website**: https://joss.theoj.org/
- **JOSS Review Criteria**: https://joss.readthedocs.io/en/latest/review_criteria.html
- **JOSS Paper Template**: https://joss.readthedocs.io/en/latest/submitting.html#example-paper-and-bibliography
- **JOSS Author Guidelines**: https://joss.readthedocs.io/en/latest/submitting.html
- **Citation File Format**: https://citation-file-format.github.io/

## Next Immediate Steps

1. Create **paper.bib** bibliography file
2. Write **paper content** (Summary, Statement of Need, etc.)
3. Verify/obtain **valid ORCID IDs** for all authors
4. Create **CONTRIBUTING.md** and **CODE_OF_CONDUCT.md**
5. Create **CITATION.cff** file
6. Improve **README.md** with comprehensive documentation
7. Create a **tagged release** and link to Zenodo
8. Validate paper locally using Docker
9. Submit to JOSS
