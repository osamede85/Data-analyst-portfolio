library(shiny)
library(randomForest)
library(shinythemes)
library(dplyr)

# Function to calculate age_status, avg_glucose_level_status, and bmi_status
calculate_status <- function(age, avg_glucose_level, bmi) {
  # Example logic for statuses; adjust according to your original dataset
  age_status <- ifelse(age < 18, 0, ifelse(age < 45, 1, ifelse(age < 60, 2, 3)))
  avg_glucose_level_status <- ifelse(avg_glucose_level < 4, 0, ifelse(avg_glucose_level < 7, 1, 2))
  bmi_status <- ifelse(bmi < 18.5, 0, ifelse(bmi < 25, 1, ifelse(bmi < 30, 2, 3)))
  
  return(list(age_status = age_status,
              avg_glucose_level_status = avg_glucose_level_status,
              bmi_status = bmi_status))
}

# Define UI
ui <- fluidPage(
  theme = shinytheme("united"),
  tags$head(
    tags$style(HTML("
            .button {
                background-color: #e95420;
                color: white;
                padding: 10px 20px;
                border: none;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                width:100%;
            }
        "))
  ),
  
  navbarPage(
    "Stroke Prediction",
    tabPanel("Model",
             sidebarLayout(
               sidebarPanel(
                 tags$h3("Input:"),
                 radioButtons("gender", h4("Gender"), 
                              choices = list("MALE" = 2, "FEMALE" = 1),
                              selected = 1),
                 numericInput("age", h4("Age"), value = 30),
                 selectInput("evermarried", h4("Ever Married"), 
                             choices = list("Yes" = 2, "No" = 1), selected = 1),
                 numericInput("avg_glucose_level", h4("Average Glucose Level"), value = 5),
                 numericInput("bmi", h4("BMI"), value = 25),
                 selectInput("heart_disease", h4("Heart Disease"), 
                             choices = list("Yes" = 2, "No" = 1), selected = 1),
                 selectInput("hypertension", h4("Hypertension"), 
                             choices = list("Yes" = 2, "No" = 1), selected = 1),
                 selectInput("smoking_status", h4("Smoking Status"),
                             choices = list("Never smoked" = 1, "Formerly smoked" = 2, "Smokes" = 3),
                             selected = 1),
                 checkboxGroupInput("work_type", h4("Work Type"),
                                    choices = list("Children" = "work_typechildren",
                                                   "Gov Job" = "work_typegov_job",
                                                   "Never Worked" = "work_typenever_worked",
                                                   "Private" = "work_typeprivate",
                                                   "Self Employed" = "work_typeself_employed"),
                                    selected = "work_typeprivate"),
                 selectInput("residence_type", h4("Residence Type"),
                             choices = list("Rural" = "residence_typerural", "Urban" = "residence_typeurban"),
                             selected = "residence_typeurban"),
                 actionButton("submit", "Predict", class = "button")
               ),
               mainPanel(
                 verbatimTextOutput("predictionOutput"),
                 verbatimTextOutput("probOutput")
               )
             )
    )
  )
)

# Define server logic
server <- function(input, output) {
  observeEvent(input$submit, {
    # Ensure input values are available
    if (is.null(input$work_type) || is.null(input$residence_type)) {
      output$predictionOutput <- renderText("Please fill all inputs.")
      return(NULL)
    }
    
    # Create a data frame with work type columns
    work_types <- data.frame(work_typechildren = 0,
                             work_typegov_job = 0,
                             work_typenever_worked = 0,
                             work_typeprivate = 0,
                             work_typeself_employed = 0)
    
    # Set the selected work type(s) to 1
    if (!is.null(input$work_type)) {
      work_types[input$work_type] <- 1
    }
    
    # Create a data frame for residence type
    residence_types <- data.frame(residence_typerural = 0, residence_typeurban = 0)
    if (input$residence_type == "residence_typerural") {
      residence_types$residence_typerural <- 1
    } else {
      residence_types$residence_typeurban <- 1
    }
    
    # Calculate statuses for age, avg_glucose_level, and bmi
    statuses <- calculate_status(input$age, input$avg_glucose_level, input$bmi)
    
    # Create the new_row data frame with all the necessary inputs
    new_row <- data.frame(
      gender = as.numeric(input$gender),
      age = as.numeric(input$age),
      ever_married = as.numeric(input$evermarried),
      avg_glucose_level = as.numeric(input$avg_glucose_level),
      bmi = as.numeric(input$bmi),
      heart_disease = as.numeric(input$heart_disease),
      hypertension = as.numeric(input$hypertension),
      smoking_status = as.numeric(input$smoking_status),
      age_status = statuses$age_status,
      avg_glucose_level_status = statuses$avg_glucose_level_status,
      bmi_status = statuses$bmi_status
    )
    
    # Add work types and residence types to the new_row
    new_row <- cbind(new_row, work_types, residence_types)
    
    # Convert avg_glucose_level to reciprocal (as in your previous logic)
    new_row$avg_glucose_level <- 1 / new_row$avg_glucose_level
    
    # Make predictions using the trained model
    prediction <- predict(model, newdata = new_row, type = "prob")
    final_prediction <- predict(model, newdata = new_row)
    
    # Output the predictions
    output$predictionOutput <- renderText({
      paste("Predicted Outcome:", ifelse(final_prediction == "Yes", "Stroke", "No Stroke"))
    })
    
    output$probOutput <- renderText({
      paste("Stroke Probability:", round(prediction[2], 3), 
            ", No Stroke Probability:", round(prediction[1], 3))
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)
