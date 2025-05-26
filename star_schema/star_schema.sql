USE lao39;

DROP TABLE IF EXISTS `fact_patient`;
DROP TABLE IF EXISTS `fact_lab`;
DROP TABLE IF EXISTS `fact_procedure`;
DROP TABLE IF EXISTS `fact_diagnosis`;
DROP TABLE IF EXISTS `dim_time`;
DROP TABLE IF EXISTS `dim_patient`;
DROP TABLE IF EXISTS `dim_provider`;
DROP TABLE IF EXISTS `dim_lab`;
DROP TABLE IF EXISTS `dim_procedure`;
DROP TABLE IF EXISTS `dim_diagnosis`;

CREATE TABLE `dim_time` (
  `date_id` int NOT NULL AUTO_INCREMENT,
  `start_dt` date NOT NULL,
  `end_dt` date NOT NULL,
  `days_in_month` int NOT NULL,
  `month` varchar(10) NOT NULL,
  `year` int NOT NULL,
  `quarter` int NOT NULL,
  PRIMARY KEY (`date_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `dim_patient` (
  `patient_id` int NOT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `last_name` varchar(50) DEFAULT NULL,
  `gender` varchar(10) DEFAULT NULL,
  `dob` date DEFAULT NULL,
  `birth_month` varchar(10) DEFAULT NULL,
  `birth_year` int DEFAULT NULL,
  PRIMARY KEY (`patient_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `dim_provider` (
  `provider_id` int NOT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `last_name` varchar(50) DEFAULT NULL,
  `specialty` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`provider_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `dim_lab` (
  `lab_id` int NOT NULL,
  `cpt_code` varchar(45) NOT NULL,
  `lab_name` varchar(255) NOT NULL,
  PRIMARY KEY (`lab_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `dim_procedure` (
  `procedure_id` int NOT NULL,
  `icd10_code` varchar(50) NOT NULL,
  `proc_name` varchar(100) NOT NULL,
  `description` varchar(255) NOT NULL,
  PRIMARY KEY (`procedure_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `dim_diagnosis` (
  `diagnosis_id` int NOT NULL,
  `name` varchar(250) NOT NULL,
  `icd10_code` varchar(45) NOT NULL,
  PRIMARY KEY (`diagnosis_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `fact_patient` (
  `provider_id` int NOT NULL,
  `patient_id` int NOT NULL,
  `date_id` int NOT NULL,
  `visit_count` int NOT NULL,
  `lab_count` int NOT NULL,
  `procedure_count` int NOT NULL,
  `diagnosis_count` int NOT NULL,
  PRIMARY KEY (`provider_id`, `patient_id`, `date_id`),
  CONSTRAINT `fact_visit_fk1` FOREIGN KEY (`provider_id`) REFERENCES `dim_provider` (`provider_id`),
  CONSTRAINT `fact_visit_fk2` FOREIGN KEY (`patient_id`) REFERENCES `dim_patient` (`patient_id`),
  CONSTRAINT `fact_visit_fk3` FOREIGN KEY (`date_id`) REFERENCES `dim_time` (`date_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `fact_lab` (
  `provider_id` int NOT NULL,
  `lab_id` int NOT NULL,
  `date_id` int NOT NULL,
  `patient_count` int NOT NULL,
  `visit_count` int NOT NULL,
  `lab_count` int NOT NULL,
  PRIMARY KEY (`provider_id`, `lab_id`, `date_id`),
  CONSTRAINT `fact_lab_fk1` FOREIGN KEY (`provider_id`) REFERENCES `dim_provider` (`provider_id`),
  CONSTRAINT `fact_lab_fk2` FOREIGN KEY (`lab_id`) REFERENCES `dim_lab` (`lab_id`),
  CONSTRAINT `fact_lab_fk3` FOREIGN KEY (`date_id`) REFERENCES `dim_time` (`date_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `fact_procedure` (
  `provider_id` int NOT NULL,
  `procedure_id` int NOT NULL,
  `date_id` int NOT NULL,
  `patient_count` int NOT NULL,
  `visit_count` int NOT NULL,
  `procedure_count` int NOT NULL,
  PRIMARY KEY (`provider_id`, `procedure_id`, `date_id`),
  CONSTRAINT `fact_procedure_fk1` FOREIGN KEY (`provider_id`) REFERENCES `dim_provider` (`provider_id`),
  CONSTRAINT `fact_procedure_fk2` FOREIGN KEY (`procedure_id`) REFERENCES `dim_procedure` (`procedure_id`),
  CONSTRAINT `fact_procedure_fk3` FOREIGN KEY (`date_id`) REFERENCES `dim_time` (`date_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `fact_diagnosis` (
  `provider_id` int NOT NULL,
  `diagnosis_id` int NOT NULL,
  `date_id` int NOT NULL,
  `patient_count` int NOT NULL,
  `visit_count` int NOT NULL,
  `diagnosis_frequency` int NOT NULL,
  PRIMARY KEY (`provider_id`, `diagnosis_id`, `date_id`),
  CONSTRAINT `fact_diagnosis_fk1` FOREIGN KEY (`provider_id`) REFERENCES `dim_provider` (`provider_id`),
  CONSTRAINT `fact_diagnosis_fk2` FOREIGN KEY (`diagnosis_id`) REFERENCES `dim_diagnosis` (`diagnosis_id`),
  CONSTRAINT `fact_diagnosis_fk3` FOREIGN KEY (`date_id`) REFERENCES `dim_time` (`date_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;