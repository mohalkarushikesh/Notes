**Playwright BDD project with Maven** — but the key detail is that Maven is a **Java build tool**, while Playwright’s primary SDK is in **Node.js** and **Python**. So the “Maven + Playwright” combination only makes sense if you’re using **Playwright for Java** (which exists as a wrapper around the Playwright engine). In that case, you can set up a project very similar to Selenium + Maven + TestNG, but with Playwright libraries instead.

---

## 1. Prerequisites
- Install **Java JDK** (11+).
- Install **Apache Maven** (`mvn -v` should work).
- IDE (IntelliJ IDEA or Eclipse).

---

## 2. Create Maven Project
```bash
mvn archetype:generate -DgroupId=com.playwright.project -DartifactId=playwright-bdd-demo -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
```

---

## 3. Project Structure
```
playwright-bdd-demo/
│
├── pom.xml
├── testng.xml
│
├── src/
│   ├── main/java/com/playwright/project/
│   │   ├── App.java
│   │   ├── pages/
│   │   │   ├── BasePage.java
│   │   │   ├── LoginPage.java
│   │   │   └── DashboardPage.java
│   │   └── utils/
│   │       └── DriverFactory.java
│   │
│   └── test/java/com/playwright/project/
│       ├── LoginTest.java
│       └── DashboardTest.java
│
└── resources/
    └── config.properties
```

---

## 4. `pom.xml` Dependencies
```xml
<dependencies>
    <!-- Playwright for Java -->
    <dependency>
        <groupId>com.microsoft.playwright</groupId>
        <artifactId>playwright</artifactId>
        <version>1.45.0</version>
    </dependency>

    <!-- TestNG -->
    <dependency>
        <groupId>org.testng</groupId>
        <artifactId>testng</artifactId>
        <version>7.10.2</version>
        <scope>test</scope>
    </dependency>

    <!-- Cucumber (for BDD) -->
    <dependency>
        <groupId>io.cucumber</groupId>
        <artifactId>cucumber-java</artifactId>
        <version>7.15.0</version>
    </dependency>
    <dependency>
        <groupId>io.cucumber</groupId>
        <artifactId>cucumber-testng</artifactId>
        <version>7.15.0</version>
    </dependency>
</dependencies>
```

---

## 5. Feature File
`src/test/resources/features/login.feature`
```gherkin
Feature: Login functionality

  Scenario: Valid login
    Given I open the login page
    When I enter valid credentials
    Then I should see the dashboard
```

---

## 6. Step Definitions
`src/test/java/com/playwright/project/steps/LoginSteps.java`
```java
package com.playwright.project.steps;

import com.microsoft.playwright.*;
import io.cucumber.java.en.*;

public class LoginSteps {
    Playwright playwright;
    Browser browser;
    Page page;

    @Given("I open the login page")
    public void openLoginPage() {
        playwright = Playwright.create();
        browser = playwright.chromium().launch(new BrowserType.LaunchOptions().setHeadless(false));
        page = browser.newPage();
        page.navigate("https://example.com/login");
    }

    @When("I enter valid credentials")
    public void enterCredentials() {
        page.fill("#username", "admin");
        page.fill("#password", "password123");
        page.click("#login");
    }

    @Then("I should see the dashboard")
    public void verifyDashboard() {
        assert page.title().contains("Dashboard");
        browser.close();
        playwright.close();
    }
}
```

---

## 7. Test Runner
`src/test/java/com/playwright/project/RunCucumberTest.java`
```java
package com.playwright.project;

import io.cucumber.testng.AbstractTestNGCucumberTests;
import io.cucumber.testng.CucumberOptions;

@CucumberOptions(
    features = "src/test/resources/features",
    glue = "com.playwright.project.steps",
    plugin = {"pretty", "html:target/cucumber-report.html"}
)
public class RunCucumberTest extends AbstractTestNGCucumberTests {
}
```

---

## 8. Execution
Run from terminal:
```bash
mvn clean test
```

- Cucumber executes `.feature` files.
- Playwright drives the browser.
- Reports generated in `target/cucumber-report.html`.

---
---



**Playwright + Maven + BDD (Cucumber + TestNG)** framework, very similar to Selenium setups but leveraging Playwright’s modern browser automation.  

Would you like me to extend this into a **multi-browser setup (Chromium, Firefox, WebKit) with parallel execution** so you can see how Playwright’s strengths shine compared to Selenium?



For a **Python + Playwright + BDD (Behavior-Driven Development) end-to-end setup**, you’ll want a structure that combines Playwright for browser automation with a BDD framework like **Behave** or **pytest-bdd**. Here’s a complete project layout and setup guide:

---

## 1. Environment Setup
- Install **Python 3.9+**.
- Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate   # Linux/Mac
  venv\Scripts\activate      # Windows
  ```
- Install dependencies:
  ```bash
  pip install playwright behave pytest-bdd
  playwright install
  ```

---

## 2. Project Structure

```
playwright-bdd-demo/
│
├── features/
│   ├── login.feature
│   └── checkout.feature
│
├── steps/
│   ├── login_steps.py
│   └── checkout_steps.py
│
├── pages/
│   ├── base_page.py
│   ├── login_page.py
│   └── checkout_page.py
│
├── utils/
│   ├── driver_factory.py
│   └── config.py
│
├── reports/
│
├── requirements.txt
└── behave.ini
```

---

## 3. Feature File (BDD)
`features/login.feature`
```gherkin
Feature: Login functionality

  Scenario: Valid login
    Given I open the login page
    When I enter valid credentials
    Then I should see the dashboard
```

---

## 4. Step Definitions
`steps/login_steps.py`
```python
from behave import given, when, then
from pages.login_page import LoginPage

@given("I open the login page")
def step_open_login(context):
    context.page = context.browser.new_page()
    context.login_page = LoginPage(context.page)
    context.login_page.open()

@when("I enter valid credentials")
def step_enter_credentials(context):
    context.login_page.login("admin", "password123")

@then("I should see the dashboard")
def step_dashboard(context):
    assert "Dashboard" in context.page.title()
```

---

## 5. Page Object Model
`pages/login_page.py`
```python
class LoginPage:
    def __init__(self, page):
        self.page = page

    def open(self):
        self.page.goto("https://example.com/login")

    def login(self, user, pwd):
        self.page.fill("#username", user)
        self.page.fill("#password", pwd)
        self.page.click("#login")
```

---

## 6. Driver Factory
`utils/driver_factory.py`
```python
from playwright.sync_api import sync_playwright

def get_browser():
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    return browser, playwright
```

---

## 7. Behave Configuration
`behave.ini`
```ini
[behave]
default_tags = ~@skip
show_skipped = false
```

---

## 8. Running Tests
- Run tests with:
  ```bash
  behave
  ```
- Reports can be generated using plugins like **Allure Behave**:
  ```bash
  pip install allure-behave
  behave -f allure_behave.formatter:AllureFormatter -o reports/
  ```

---

## 9. Enhancements
- **Data-driven testing**: Use Scenario Outlines in `.feature` files.
- **Parallel execution**: Use `pytest-playwright` with `pytest-xdist`.
- **CI/CD integration**: Add to GitHub Actions or Jenkins pipelines.
- **Cross-browser testing**: Launch Chromium, Firefox, WebKit with Playwright.

---

This gives you a **complete end-to-end Python Playwright BDD framework** with modular structure, reusable page objects, and reporting.  
