package crawler;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.Keys;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxProfile;
import org.openqa.selenium.remote.DesiredCapabilities;

public class Crawler  {
    private final static String HARDIR = "/har_results";

    public static void main(String[] args) throws IOException {
        WebDriver driver = null;

        boolean hasException = false;
        boolean isAdded = false;
        ArrayList<String> webpages = new ArrayList<String>();
        Scanner s = new Scanner(new File("target_pages.txt"));
        while (s.hasNextLine()){
            webpages.add(s.nextLine());
        }
        s.close();

        try {
            // Initialization
            String OSName = System.getProperty("os.name").toLowerCase();
            String OSType = System.getProperty("os.arch");
            if(OSName.indexOf("mac") >= 0)
              System.setProperty("webdriver.gecko.driver", System.getProperty("user.dir") + "/../selenium/geckodriver-mac");
            else if(OSName.indexOf("nux") >= 0)
              if(OSType.indexOf("64") >= 0)
                System.setProperty("webdriver.gecko.driver", System.getProperty("user.dir") + "/../selenium/geckodriver-linux64");
              else
                System.setProperty("webdriver.gecko.driver", System.getProperty("user.dir") + "/../selenium/geckodriver-linux32");

            // Navigate the target websites
            for(int i = 0; i < webpages.size(); i++){
            	hasException = false;
            	DesiredCapabilities capabilities = DesiredCapabilities.firefox();
            	FirefoxProfile profile = new FirefoxProfile();

                // Load and add the Har Export Trigger extension
                File harExport = new File("har_export_trigger-0.5.0-beta.7-fx.xpi");
                profile.addExtension(harExport);

                // Enable the automation without having a new HAR file created for every loaded page.
                profile.setPreference("extensions.netmonitor.har.enableAutomation", false);
                // Set to a token that is consequently passed into all HAR API calls to verify the user.
                profile.setPreference("extensions.netmonitor.har.contentAPIToken", "test");
                // Set if you want to have the HAR object available without the developer toolbox being open.
                profile.setPreference("extensions.netmonitor.har.autoConnect", false);

                // Enable netmonitor
                profile.setPreference("devtools.netmonitor.enabled", true);
                // If set to true the final HAR file is zipped. This might represents great disk-space optimization especially if HTTP response bodies are included.
                profile.setPreference("devtools.netmonitor.har.compress", false);
                // Default log directory for generate HAR files. If empty all automatically generated HAR files are stored in <FF-profile>/har/logs
                profile.setPreference("devtools.netmonitor.har.defaultLogDir", HARDIR);
                // If true, a new HAR file is created for every loaded page automatically.
                profile.setPreference("devtools.netmonitor.har.enableAutoExportToFile", true);
                // The result HAR file is created even if there are no HTTP requests.
                profile.setPreference("devtools.netmonitor.har.forceExport", true);
                // If set to true, HTTP response bodies are also included in the HAR file (can produce significantly bigger amount of data).
                profile.setPreference("devtools.netmonitor.har.includeResponseBodies", false);
                // If set to true the export format is HARP (support for JSONP syntax that is easily transferable cross domains)
                profile.setPreference("devtools.netmonitor.har.jsonp", false);
                // Default name of JSONP callback (used for HARP format)
                profile.setPreference("devtools.netmonitor.har.jsonpCallback", false);
                // Amount of time [ms] the auto-exporter should wait after the last finished request before exporting the HAR file.
                profile.setPreference("devtools.netmonitor.har.pageLoadedTimeout", "2500");

                final File harDir = new File(HARDIR);
                int numFiles = harDir.listFiles().length;

            	String webpage = webpages.get(i);

            	String webpage_url = "http://" + webpage;

            	// Default name of the target HAR file. The default file name supports formatters
                profile.setPreference("devtools.netmonitor.har.defaultFileName", webpage);

                capabilities.setCapability(FirefoxDriver.PROFILE, profile);

            	driver = new FirefoxDriver(capabilities);

            	// Print the number of files before adding a HAR file
            	numFiles = harDir.listFiles().length;
            	System.out.println(harDir.getPath() + ": " + numFiles);

            	driver.get("about:preferences");
            	driver.findElement(By.tagName("page")).sendKeys(Keys.SHIFT, Keys.F5);
            	Thread.sleep(2000L);

            	try {
                    //driver.manage().timeouts().implicitlyWait(210, TimeUnit.SECONDS);
            		driver.manage().timeouts().pageLoadTimeout(210, TimeUnit.SECONDS);
            		System.out.println("Before Get...");
                	driver.get(webpage_url);
                	System.out.println("After Get...");
            	}
            	catch(TimeoutException timeException){
            		FileWriter fw = new FileWriter("NotLoaded.txt", true);
            		fw.write(webpage_url + "\n");
            		fw.close();
            		hasException = true;
                    System.err.println(timeException.toString());
            		driver.quit();
            	}
            	catch (Exception notFoundException){
            		FileWriter fw = new FileWriter("NotFound.txt", true);
            		fw.write(webpage_url + "\n");
            		fw.close();
            		hasException = true;
            		System.err.println("Not Found Exception");
            		driver.quit();
            	}

                // Wait for the new HAR file
            	isAdded = false;
                for (int c = 0; c < 90; c++) {
                	System.out.println("Preparing the HAR file..." + c);
                	if (harDir.listFiles().length > numFiles) {
                		isAdded = true;
                		System.out.println("added");
                		break;
                	}
                	Thread.sleep(1000L);
                }
                if(!isAdded && !hasException) {
                	FileWriter fw = new FileWriter("NotLoaded.txt", true);
            		fw.write(webpage_url + "\n");
            		fw.close();
                }
                
                // Print tje number of files after adding a HAR file
                System.out.println(harDir.getPath() + ": " + harDir.listFiles().length);
                if (driver != null) {
                    driver.quit();
                }
            }
        }
        catch (Exception exc) {
            System.err.println(exc.toString());
        }
    }
}
