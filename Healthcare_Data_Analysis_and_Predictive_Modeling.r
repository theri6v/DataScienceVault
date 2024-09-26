{
  "metadata": {
    "kernelspec": {
      "name": "ir",
      "display_name": "R",
      "language": "R"
    },
    "language_info": {
      "name": "R",
      "codemirror_mode": "r",
      "pygments_lexer": "r",
      "mimetype": "text/x-r-source",
      "file_extension": ".r",
      "version": "4.4.0"
    },
    "kaggle": {
      "accelerator": "none",
      "dataSources": [
        {
          "sourceId": 6213404,
          "sourceType": "datasetVersion",
          "datasetId": 3567993
        }
      ],
      "dockerImageVersionId": 30749,
      "isInternetEnabled": true,
      "language": "r",
      "sourceType": "notebook",
      "isGpuEnabled": false
    },
    "colab": {
      "name": "Healthcare Data Analysis and Predictive Modeling",
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/theri6v/DataScienceVault/blob/main/Healthcare_Data_Analysis_and_Predictive_Modeling.r\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "source": [
        "\n",
        "# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES\n",
        "# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,\n",
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S R\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "\n",
        "DATA_SOURCE_MAPPING = 'pima-indians-diabetes-database:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F3567993%2F6213404%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240926%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240926T124450Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D02c000f38a46dc80f32d9e34a258c1ff547f92ef9e81fb84a6ca46c28717dc8b1391f892085edd154674e0bb2ae36faa57853b4713cacdfa60fe1177ce5479a4c389b8ac82f35bcf20fd5d89d0196d044be47c8abb8f66877e81dcab04ac1efafb9687262c675e7731820b3863fbd2744e4119547e31c5be199b029c5f4fb3f1c343b7619be0f7b5d6be91917ce9c9d893a5937c49095fb3c846524c01e3a97d286f70e51795948f873076df61c7b3738702a0414e4a816383ed21823b29a4b8f0f0d8d1a6dd0118bd7579e4a178e39732c7cab8991fb2ab3b7cacfa5cf428889cf327463a9ef26f8e285931ac58ebfbf28fb5ab51b4d3e8e20536c4670d9c66'\n",
        "\n",
        "KAGGLE_INPUT_PATH = '/kaggle/input'\n",
        "KAGGLE_WORKING_PATH = '/kaggle/working'\n",
        "\n",
        "system(paste0('sudo umount ', '/kaggle/input'))\n",
        "system(paste0('sudo rmdir ', '/kaggle/input'))\n",
        "system(paste0('sudo mkdir -p -- ', KAGGLE_INPUT_PATH), intern=TRUE)\n",
        "system(paste0('sudo chmod 777 ', KAGGLE_INPUT_PATH), intern=TRUE)\n",
        "system(\n",
        "  paste0('sudo ln -sfn ', KAGGLE_INPUT_PATH,' ',file.path('..', 'input')),\n",
        "  intern=TRUE)\n",
        "\n",
        "system(paste0('sudo mkdir -p -- ', KAGGLE_WORKING_PATH), intern=TRUE)\n",
        "system(paste0('sudo chmod 777 ', KAGGLE_WORKING_PATH), intern=TRUE)\n",
        "system(\n",
        "  paste0('sudo ln -sfn ', KAGGLE_WORKING_PATH, ' ', file.path('..', 'working')),\n",
        "  intern=TRUE)\n",
        "\n",
        "data_source_mappings = strsplit(DATA_SOURCE_MAPPING, ',')[[1]]\n",
        "for (data_source_mapping in data_source_mappings) {\n",
        "    path_and_url = strsplit(data_source_mapping, ':')\n",
        "    directory = path_and_url[[1]][1]\n",
        "    download_url = URLdecode(path_and_url[[1]][2])\n",
        "    filename = sub(\"\\\\?.+\", \"\", download_url)\n",
        "    destination_path = file.path(KAGGLE_INPUT_PATH, directory)\n",
        "    print(paste0('Downloading and uncompressing: ', directory))\n",
        "    if (endsWith(filename, '.zip')){\n",
        "      temp = tempfile(fileext = '.zip')\n",
        "      download.file(download_url, temp)\n",
        "      unzip(temp, overwrite = TRUE, exdir = destination_path)\n",
        "      unlink(temp)\n",
        "    }\n",
        "    else{\n",
        "      temp = tempfile(fileext = '.tar')\n",
        "      download.file(download_url, temp)\n",
        "      untar(temp, exdir = destination_path)\n",
        "      unlink(temp)\n",
        "    }\n",
        "    print(paste0('Downloaded and uncompressed: ', directory))\n",
        "}\n",
        "\n",
        "print(paste0('Data source import complete'))\n"
      ],
      "metadata": {
        "id": "LskwhDI_XSFh"
      },
      "cell_type": "code",
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "# libraries\n",
        "\n",
        "library(ggplot2)      # For visualizations\n",
        "library(caret)        # For modeling and evaluation\n",
        "library(dplyr)        # For data manipulation\n",
        "library(e1071)        # For SVM model\n",
        "library(randomForest) # For Random Forest model\n",
        "library(pROC)         # For ROC curves\n",
        "library(corrplot)     # For correlation matrix\n",
        "library(ROSE)         # For oversampling techniques"
      ],
      "metadata": {
        "_uuid": "051d70d956493feee0c6d64651c6a088724dca2a",
        "_execution_state": "idle",
        "execution": {
          "iopub.status.busy": "2024-09-26T12:20:24.915936Z",
          "iopub.execute_input": "2024-09-26T12:20:24.918168Z",
          "iopub.status.idle": "2024-09-26T12:20:24.948133Z"
        },
        "trusted": true,
        "id": "NSL2BIXdXSFl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Import the Dataset"
      ],
      "metadata": {
        "id": "mobF5MXJXSFl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the dataset\n",
        "data <- read.csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:47:02.781385Z",
          "iopub.execute_input": "2024-09-26T11:47:02.783394Z",
          "iopub.status.idle": "2024-09-26T11:47:02.815732Z"
        },
        "trusted": true,
        "id": "SMuuvcalXSFm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Preview the data\n",
        "head(data)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:47:25.881538Z",
          "iopub.execute_input": "2024-09-26T11:47:25.883299Z",
          "iopub.status.idle": "2024-09-26T11:47:25.914554Z"
        },
        "trusted": true,
        "id": "X94t1rVBXSFm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "summary(data)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:47:20.539783Z",
          "iopub.execute_input": "2024-09-26T11:47:20.541551Z",
          "iopub.status.idle": "2024-09-26T11:47:20.562976Z"
        },
        "trusted": true,
        "id": "fLDgT-uBXSFm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Check for any missing values\n",
        "missing_values <- sum(is.na(data))\n",
        "print(paste(\"Total missing values: \", missing_values))"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:57:49.815291Z",
          "iopub.execute_input": "2024-09-26T11:57:49.817176Z",
          "iopub.status.idle": "2024-09-26T11:57:49.837318Z"
        },
        "trusted": true,
        "id": "dd2BVnhEXSFm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Scale Numerical Features"
      ],
      "metadata": {
        "id": "QRKF75IfXSFm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "yzG6KiQKXSFm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Feature scaling: Scale the numerical features\n",
        "data_scaled <- data %>%\n",
        "  mutate(across(where(is.numeric), scale))\n",
        "\n",
        "# Convert the outcome to a factor (binary classification)\n",
        "data_scaled$Outcome <- as.factor(data_scaled$Outcome)\n",
        "\n",
        "# Split the data into train and test sets\n",
        "set.seed(123)\n",
        "train_index <- createDataPartition(data_scaled$Outcome, p=0.7, list=FALSE)\n",
        "train_data <- data_scaled[train_index, ]\n",
        "test_data <- data_scaled[-train_index, ]"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:41:27.008811Z",
          "iopub.execute_input": "2024-09-26T11:41:27.010656Z",
          "iopub.status.idle": "2024-09-26T11:41:27.088008Z"
        },
        "trusted": true,
        "id": "u2dwZUvpXSFn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Matrix Visualization"
      ],
      "metadata": {
        "id": "3KggaIAEXSFn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Correlation matrix visualization\n",
        "corr_matrix <- cor(data_scaled %>% select(-Outcome))\n",
        "corrplot(corr_matrix, method = \"circle\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:44:03.038153Z",
          "iopub.execute_input": "2024-09-26T11:44:03.03991Z",
          "iopub.status.idle": "2024-09-26T11:44:03.177226Z"
        },
        "trusted": true,
        "id": "fRD9zBYnXSFn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Outcome distribution\n",
        "ggplot(data_scaled, aes(x=Outcome, fill=Outcome)) +\n",
        "  geom_bar() +\n",
        "  labs(title=\"Outcome Distribution\", x=\"Diabetes Outcome\", y=\"Count\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:43:38.983843Z",
          "iopub.execute_input": "2024-09-26T11:43:38.985741Z",
          "iopub.status.idle": "2024-09-26T11:43:39.310227Z"
        },
        "trusted": true,
        "id": "HOWlWklGXSFn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Glucose level distribution\n",
        "ggplot(data_scaled, aes(x=Glucose, fill=Outcome)) +\n",
        "  geom_histogram(bins=30, alpha=0.7, position='identity') +\n",
        "  labs(title=\"Glucose Level Distribution by Outcome\", x=\"Glucose\", y=\"Count\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T11:43:35.732929Z",
          "iopub.execute_input": "2024-09-26T11:43:35.734724Z",
          "iopub.status.idle": "2024-09-26T11:43:36.038657Z"
        },
        "trusted": true,
        "id": "QzwUrwHUXSFn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Train model"
      ],
      "metadata": {
        "id": "U2-oZCOAXSFn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Train a logistic regression model\n",
        "log_model <- train(Outcome ~ ., data=train_data, method=\"glm\", family=\"binomial\")\n",
        "\n",
        "# Make predictions\n",
        "log_pred <- predict(log_model, test_data)\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:04:02.688885Z",
          "iopub.execute_input": "2024-09-26T12:04:02.690726Z",
          "iopub.status.idle": "2024-09-26T12:04:03.124661Z"
        },
        "trusted": true,
        "id": "gLR9wFP2XSFo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Modele Evaluate"
      ],
      "metadata": {
        "id": "50emn2PCXSFo"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Evaluate the model\n",
        "confusionMatrix(log_pred, test_data$Outcome)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:04:06.443655Z",
          "iopub.execute_input": "2024-09-26T12:04:06.445647Z",
          "iopub.status.idle": "2024-09-26T12:04:06.466234Z"
        },
        "trusted": true,
        "id": "LnZL10EEXSFo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Make Predictions"
      ],
      "metadata": {
        "id": "9WNHz3pwXSFo"
      }
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "2wyYmf4WXSFo"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Train a Random Forest model\n",
        "rf_model <- train(Outcome ~ ., data=train_data, method=\"rf\", importance=TRUE)\n",
        "\n",
        "# Make predictions\n",
        "rf_pred <- predict(rf_model, test_data)\n",
        "\n",
        "# Evaluate the model\n",
        "confusionMatrix(rf_pred, test_data$Outcome)\n",
        "\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:06:49.308841Z",
          "iopub.execute_input": "2024-09-26T12:06:49.310622Z",
          "iopub.status.idle": "2024-09-26T12:07:23.886441Z"
        },
        "trusted": true,
        "id": "fyEaILmSXSFo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Feature Importance"
      ],
      "metadata": {
        "id": "1XIehimmXSFo"
      }
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "eKaUP3hHXSFp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Feature importance\n",
        "varImpPlot(rf_model$finalModel)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:06:38.723437Z",
          "iopub.execute_input": "2024-09-26T12:06:38.72577Z",
          "iopub.status.idle": "2024-09-26T12:06:38.824099Z"
        },
        "trusted": true,
        "id": "xKIezSM-XSFp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# ROC Curve for Random Forest"
      ],
      "metadata": {
        "id": "5dPg6Z4QXSFp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# ROC Curve for Logistic Regression\n",
        "log_roc <- roc(test_data$Outcome, as.numeric(log_pred))\n",
        "plot(log_roc, col=\"blue\", main=\"ROC Curve - Logistic Regression\")\n",
        "\n",
        "# ROC Curve for Random Forest\n",
        "rf_roc <- roc(test_data$Outcome, as.numeric(rf_pred))\n",
        "plot(rf_roc, col=\"red\", add=TRUE)\n",
        "\n",
        "\n",
        "# Add a legend\n",
        "legend(\"bottomright\", legend=c(\"Logistic Regression\", \"Random Forest\"), col=c(\"blue\", \"red\"), lty=1)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:18:23.255046Z",
          "iopub.execute_input": "2024-09-26T12:18:23.257451Z",
          "iopub.status.idle": "2024-09-26T12:18:23.403905Z"
        },
        "trusted": true,
        "id": "AZVJmMQoXSFp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Model Performance"
      ],
      "metadata": {
        "id": "IGs_v6I5XSFp"
      }
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "tsyuQufhXSFp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Confusion Matrix for Logistic Regression\n",
        "log_cm <- confusionMatrix(log_pred, test_data$Outcome)\n",
        "print(log_cm)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:15:42.071696Z",
          "iopub.execute_input": "2024-09-26T12:15:42.073523Z",
          "iopub.status.idle": "2024-09-26T12:15:42.095621Z"
        },
        "trusted": true,
        "id": "hEHW3ilnXSFp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Confusion Matrix for Random Forest\n",
        "rf_cm <- confusionMatrix(rf_pred, test_data$Outcome)\n",
        "print(rf_cm)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:13:51.794435Z",
          "iopub.execute_input": "2024-09-26T12:13:51.79606Z",
          "iopub.status.idle": "2024-09-26T12:13:51.815969Z"
        },
        "trusted": true,
        "id": "5-Gdij0kXSFp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Calculate Precision, Recall, and F1 Score"
      ],
      "metadata": {
        "id": "WhpHUsfJXSFp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# For Logistic Regression\n",
        "log_precision <- posPredValue(log_pred, test_data$Outcome, positive = \"1\")\n",
        "log_recall <- sensitivity(log_pred, test_data$Outcome, positive = \"1\")\n",
        "log_f1 <- (2 * log_precision * log_recall) / (log_precision + log_recall)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:16:11.918403Z",
          "iopub.execute_input": "2024-09-26T12:16:11.920309Z",
          "iopub.status.idle": "2024-09-26T12:16:11.939658Z"
        },
        "trusted": true,
        "id": "TevHRe67XSFq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# For Random Forest\n",
        "rf_precision <- posPredValue(rf_pred, test_data$Outcome, positive = \"1\")\n",
        "rf_recall <- sensitivity(rf_pred, test_data$Outcome, positive = \"1\")\n",
        "rf_f1 <- (2 * rf_precision * rf_recall) / (rf_precision + rf_recall)\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:16:13.974876Z",
          "iopub.execute_input": "2024-09-26T12:16:13.97683Z",
          "iopub.status.idle": "2024-09-26T12:16:14.005146Z"
        },
        "trusted": true,
        "id": "yOW8DXAHXSFq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Display the metrics\n",
        "cat(\"Logistic Regression - Precision:\", log_precision, \"Recall:\", log_recall, \"F1 Score:\", log_f1, \"\\n\")\n",
        "cat(\"Random Forest - Precision:\", rf_precision, \"Recall:\", rf_recall, \"F1 Score:\", rf_f1, \"\\n\")\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:13:59.465816Z",
          "iopub.execute_input": "2024-09-26T12:13:59.467477Z",
          "iopub.status.idle": "2024-09-26T12:13:59.4868Z"
        },
        "trusted": true,
        "id": "_NIeAoNfXSFq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Cross-Validation"
      ],
      "metadata": {
        "id": "N8LJaAV2XSFq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "set.seed(123)\n",
        "cv_control <- trainControl(method = \"cv\", number = 10)  # 10-fold cross-validation\n",
        "\n",
        "# Train the Logistic Regression model with cross-validation\n",
        "log_cv_model <- train(Outcome ~ ., data = train_data, method = \"glm\", family = \"binomial\", trControl = cv_control)\n",
        "\n",
        "# Train the Random Forest model with cross-validation\n",
        "rf_cv_model <- train(Outcome ~ ., data = train_data, method = \"rf\", trControl = cv_control)\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:14:54.52045Z",
          "iopub.execute_input": "2024-09-26T12:14:54.522238Z",
          "iopub.status.idle": "2024-09-26T12:15:03.828242Z"
        },
        "trusted": true,
        "id": "pJdGuKyWXSFq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Compare cross-validated results\n",
        "cv_results <- resamples(list(logistic = log_cv_model, rf = rf_cv_model))\n",
        "summary(cv_results)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:16:46.353517Z",
          "iopub.execute_input": "2024-09-26T12:16:46.355394Z",
          "iopub.status.idle": "2024-09-26T12:16:46.389286Z"
        },
        "trusted": true,
        "id": "D1la3y8nXSFr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dotplot(cv_results, main = \"Cross-Validated Model Comparison\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-09-26T12:16:39.341648Z",
          "iopub.execute_input": "2024-09-26T12:16:39.343565Z",
          "iopub.status.idle": "2024-09-26T12:16:39.522525Z"
        },
        "trusted": true,
        "id": "BegTgn9xXSFr"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}