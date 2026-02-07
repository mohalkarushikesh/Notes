**complete end-to-end Selenium project setup with Maven using a BDD approach**, you’ll combine Selenium WebDriver with **Cucumber** (for BDD) and **TestNG/JUnit** (for execution). Let’s lay out the full structure and flow:

---

## 1. Prerequisites
- **Java JDK** (11+ recommended).
- **Apache Maven** installed and configured (`mvn -v` works).
- **IDE** (IntelliJ IDEA, Eclipse).
- **Browser Driver** (e.g., ChromeDriver).

---

## 2. Project Structure

```
selenium-bdd-demo/
│
├── pom.xml
├── testng.xml
│
├── src/
│   ├── main/java/com/selenium/project/
│   │   ├── pages/
│   │   │   ├── BasePage.java
│   │   │   ├── LoginPage.java
│   │   │   └── DashboardPage.java
│   │   └── utils/
│   │       ├── DriverFactory.java
│   │       └── ConfigReader.java
│   │
│   └── test/java/com/selenium/project/
│       ├── steps/
│       │   └── LoginSteps.java
│       ├── runners/
│       │   └── TestRunner.java
│       └── hooks/
│           └── Hooks.java
│
├── src/test/resources/
│   └── features/
│       └── login.feature
│
├── drivers/
│   └── chromedriver.exe
│
└── reports/
    └── cucumber-report.html
```

---

## 3. `pom.xml` Dependencies

```xml
<dependencies>
    <!-- Selenium -->
    <dependency>
        <groupId>org.seleniumhq.selenium</groupId>
        <artifactId>selenium-java</artifactId>
        <version>4.21.0</version>
    </dependency>

    <!-- Cucumber for BDD -->
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

    <!-- TestNG -->
    <dependency>
        <groupId>org.testng</groupId>
        <artifactId>testng</artifactId>
        <version>7.10.2</version>
        <scope>test</scope>
    </dependency>
</dependencies>

<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-surefire-plugin</artifactId>
            <version>3.2.5</version>
            <configuration>
                <suiteXmlFiles>
                    <suiteXmlFile>testng.xml</suiteXmlFile>
                </suiteXmlFiles>
            </configuration>
        </plugin>
    </plugins>
</build>
```

---

## 4. Feature File
`src/test/resources/features/login.feature`

```gherkin
Feature: Login functionality

  Scenario: Valid login
    Given I open the login page
    When I enter valid credentials
    Then I should see the dashboard
```

---

## 5. Step Definitions
`src/test/java/com/selenium/project/steps/LoginSteps.java`

```java
package com.selenium.project.steps;

import com.selenium.project.pages.LoginPage;
import io.cucumber.java.en.*;
import org.testng.Assert;

public class LoginSteps {
    LoginPage loginPage;

    @Given("I open the login page")
    public void openLoginPage() {
        loginPage = new LoginPage();
        loginPage.open();
    }

    @When("I enter valid credentials")
    public void enterCredentials() {
        loginPage.login("admin", "password123");
    }

    @Then("I should see the dashboard")
    public void verifyDashboard() {
        Assert.assertTrue(loginPage.isDashboardVisible());
    }
}
```

---

## 6. Page Object Example
`src/main/java/com/selenium/project/pages/LoginPage.java`

```java
package com.selenium.project.pages;

import com.selenium.project.utils.DriverFactory;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class LoginPage {
    WebDriver driver;

    public LoginPage() {
        driver = DriverFactory.getDriver();
    }

    public void open() {
        driver.get("https://example.com/login");
    }

    public void login(String user, String pwd) {
        driver.findElement(By.id("username")).sendKeys(user);
        driver.findElement(By.id("password")).sendKeys(pwd);
        driver.findElement(By.id("login")).click();
    }

    public boolean isDashboardVisible() {
        return driver.getTitle().contains("Dashboard");
    }
}
```

---

## 7. Hooks (Setup/Teardown)
`src/test/java/com/selenium/project/hooks/Hooks.java`

```java
package com.selenium.project.hooks;

import com.selenium.project.utils.DriverFactory;
import io.cucumber.java.After;
import io.cucumber.java.Before;

public class Hooks {
    @Before
    public void setup() {
        DriverFactory.initDriver();
    }

    @After
    public void teardown() {
        DriverFactory.quitDriver();
    }
}
```

---

## 8. Runner Class
`src/test/java/com/selenium/project/runners/TestRunner.java`

```java
package com.selenium.project.runners;

import io.cucumber.testng.AbstractTestNGCucumberTests;
import io.cucumber.testng.CucumberOptions;

@CucumberOptions(
    features = "src/test/resources/features",
    glue = {"com.selenium.project.steps", "com.selenium.project.hooks"},
    plugin = {"pretty", "html:reports/cucumber-report.html"}
)
public class TestRunner extends AbstractTestNGCucumberTests {
}
```

---

## 9. Execution
Run from terminal:

```bash
mvn clean test
```

- Cucumber executes `.feature` files.
- Selenium drives the browser.
- Reports generated in `reports/cucumber-report.html`.

---

## 10. Enhancements
- **Data-driven scenarios**: Use Scenario Outlines in `.feature` files.
- **Parallel execution**: Configure TestNG or Cucumber parallel options.
- **Advanced reporting**: Integrate Allure or Extent Reports.
- **CI/CD**: Add Jenkins or GitHub Actions pipeline.

---

This gives you a **complete Maven + Selenium + Cucumber (BDD) project setup** with installation, configuration, and execution.  
