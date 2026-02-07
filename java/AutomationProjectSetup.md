**complete end-to-end Selenium project setup with Maven**, including installation, configuration, and execution.

---

## 1. Prerequisites
- **Java JDK** installed (Java 8+ recommended).
- **Apache Maven** installed and added to PATH (`mvn -v` should work).
  - To set up **Apache Maven** and configure your environment properly, follow these steps in detail:
  
  ---
  
  ### 1. Download Maven
  - Go to the official Apache Maven site: `https://maven.apache.org/download.cgi` [(maven.apache.org in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fmaven.apache.org%2Fdownload.cgi").
  - Download the **binary zip archive** (e.g., `apache-maven-3.9.x-bin.zip`).
  
  ---
  
  ### 2. Extract Maven
  - Unzip the archive to a location on your system, for example:
    - **Windows**: `C:\Program Files\Apache\Maven`
    - **Linux/Mac**: `/opt/apache-maven`
  
  ---
  
  ### 3. Set Environment Variables
  ### Windows
  1. Open **System Properties → Advanced → Environment Variables**.
  2. Add a new **System Variable**:
     - **Variable name**: `MAVEN_HOME`
     - **Variable value**: `C:\Program Files\Apache\Maven\apache-maven-3.9.x`
  3. Edit the **Path** variable and add:
     ```
     %MAVEN_HOME%\bin
     ```
  4. Click OK to save.
  
  #### Linux / macOS
  Edit your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.profile`) and add:
  
  ```bash
  export MAVEN_HOME=/opt/apache-maven/apache-maven-3.9.x
  export PATH=$MAVEN_HOME/bin:$PATH
  ```
  
  Then reload:
  ```bash
  source ~/.bashrc
  ```
  
  ---
  
  ### 4. Verify Installation
  Run in terminal/command prompt:
  
  ```bash
  mvn -v
  ```
  
  Expected output:
  ```
  Apache Maven 3.9.x
  Maven home: C:\Program Files\Apache\Maven\apache-maven-3.9.x
  Java version: 11.0.x, vendor: Oracle Corporation
  Default locale: en_US, platform encoding: UTF-8
  OS name: Windows 10, version: 10.0, arch: amd64
  ```
  
  ---
  
  ### 5. Connect Maven with Your Project
  - Ensure your project has a `pom.xml` file at the root.
  - Run:
    ```bash
    mvn clean install
    ```
    This compiles, tests, and packages your project.
  
  ---
  
  ### 6. IDE Integration
  - **IntelliJ IDEA**: Maven support is built-in. Just open the project folder containing `pom.xml`.
  - **Eclipse**: Install the **Maven Integration (m2e)** plugin, then import the project as a Maven project.
  
  ---
  
  Once Maven is installed and configured, you can run Selenium tests with:
  
  ```bash
  mvn test
  ```
  
  This will execute your TestNG suite defined in `testng.xml`.

---

- **IDE** (IntelliJ IDEA, Eclipse, or VS Code).
- **Browser Driver** (e.g., ChromeDriver).

---

## 2. Create Maven Project
From terminal:

```bash
mvn archetype:generate -DgroupId=com.selenium.project -DartifactId=selenium-demo -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
```

This generates a project structure:

A **full Maven + Selenium + TestNG project structure** that’s ready for real-world automation looks like this:

```
selenium-demo/
│
├── pom.xml
├── testng.xml
│
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── selenium/
│   │               └── project/
│   │                   ├── App.java
│   │                   ├── utils/
│   │                   │   ├── DriverFactory.java
│   │                   │   └── ConfigReader.java
│   │                   └── pages/
│   │                       ├── BasePage.java
│   │                       ├── LoginPage.java
│   │                       └── DashboardPage.java
│   │
│   └── test/
│       └── java/
│           └── com/
│               └── selenium/
│                   └── project/
│                       ├── AppTest.java
│                       ├── LoginTest.java
│                       └── DashboardTest.java
│
├── drivers/
│   └── chromedriver.exe   (or chromedriver for Linux/Mac)
│
├── reports/
│   └── test-output/       (generated after execution)
│
└── resources/
    └── config.properties
```

---

### Key Components
- **pom.xml** → Maven dependencies (Selenium, TestNG, Surefire plugin).
- **testng.xml** → Defines test suite and classes to run.
- **src/main/java/com/selenium/project/pages/** → Page Object Model classes (LoginPage, DashboardPage).
- **src/main/java/com/selenium/project/utils/** → Utility classes (DriverFactory for WebDriver setup, ConfigReader for properties).
- **src/test/java/com/selenium/project/** → Test classes (LoginTest, DashboardTest).
- **drivers/** → Browser drivers.
- **resources/config.properties** → Config file for URLs, credentials, environment variables.
- **reports/** → Test execution reports (Surefire, Allure, Extent).

---

### Example Flow
1. **DriverFactory.java** sets up Chrome/Firefox driver.
2. **LoginPage.java** contains locators and methods for login.
3. **LoginTest.java** calls `LoginPage.login()` and asserts results.
4. **testng.xml** runs all tests.
5. **mvn clean test** executes suite and generates reports.

---

This structure is modular, scalable, and CI/CD friendly. You can easily extend it with:
- **Data-driven testing** (Excel/JSON).
- **Parallel execution** (TestNG parallel tests).
- **Advanced reporting** (Allure/Extent).

---

## 3. Configure `pom.xml`
Add Selenium and TestNG dependencies:

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.selenium.project</groupId>
    <artifactId>selenium-demo</artifactId>
    <version>1.0-SNAPSHOT</version>

    <dependencies>
        <!-- Selenium -->
        <dependency>
            <groupId>org.seleniumhq.selenium</groupId>
            <artifactId>selenium-java</artifactId>
            <version>4.21.0</version>
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
            <!-- Maven Surefire Plugin for running TestNG -->
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
</project>
```

---

## 4. Create TestNG Suite (`testng.xml`)
At project root:

```xml
<!DOCTYPE suite SYSTEM "https://testng.org/testng-1.0.dtd" >
<suite name="SeleniumSuite">
    <test name="LoginTests">
        <classes>
            <class name="com.selenium.project.LoginTest"/>
        </classes>
    </test>
</suite>
```

---

## 5. Write a Sample Test
`src/test/java/com/selenium/project/LoginTest.java`

```java
package com.selenium.project;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.*;

public class LoginTest {
    WebDriver driver;

    @BeforeClass
    public void setup() {
        System.setProperty("webdriver.chrome.driver", "drivers/chromedriver.exe");
        driver = new ChromeDriver();
        driver.manage().window().maximize();
    }

    @Test
    public void testValidLogin() {
        driver.get("https://example.com/login");
        driver.findElement(By.id("username")).sendKeys("admin");
        driver.findElement(By.id("password")).sendKeys("password123");
        driver.findElement(By.id("login")).click();
        Assert.assertTrue(driver.getTitle().contains("Dashboard"));
    }

    @AfterClass
    public void teardown() {
        if (driver != null) {
            driver.quit();
        }
    }
}
```

---

## 6. Execute Tests
Run from terminal:

```bash
mvn clean test
```

- Maven downloads dependencies.
- TestNG executes tests defined in `testng.xml`.
- Results appear in console and `target/surefire-reports`.

---

## 7. Enhancements
- **Page Object Model (POM)**: Create `pages/` classes for maintainability.
- **Data-driven testing**: Integrate Apache POI or JSON for test data.
- **Reports**: Add **Allure** or **Extent Reports** for rich HTML reports.
- **CI/CD**: Integrate with Jenkins, GitHub Actions, or Azure DevOps.

---

This setup gives you a **fully functional Maven + Selenium + TestNG project** that you can run locally or plug into CI/CD pipelines.  
