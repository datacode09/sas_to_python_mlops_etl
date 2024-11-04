/* Set global SAS options */
options MCOMPILENONE=NOENNING;    /* Disables macro compilation warnings */
options varlenchr=nomain;         /* Sets variable length behavior for character variables */

/* Include required macros */
%include macros;                  /* Includes macro definitions */

/* Define file paths for the system */
%let ppath = /sasdata/nacb/dev3/cb_production/100_arm_models/8002_Folder_Structure/00_SAS_Codes/;    /* Path for SAS code files */
%let cpath = /sasdata/nacb/dev3/cb_production/100_arm_models/universe_customer/code/;                /* Path for customer code */
%include "&ppath.800_Score_launcher_setup_v9.sas";                                                   /* Include setup configuration */

/* Define input/output paths */
libname inlib '/sasdata/nacb/dev3/cb_production/100_arm_models/8002_Folder_Structure/01_Input_Data/' access=readonly;  /* Input library with read-only access */
libname outlib '/sasdata/nacb/dev3/cb_production/100_arm_models/8002_Folder_Structure/02_Output_Data/';                /* Output library for results */

/* Include data processing scripts */
/* %include "&ppath.600_Path-data-append.sas"; */    /* Commented out data append script */
/* %include "&ppath.INLRGE_PRC_MOD_v1_patch.sas"; */ /* Commented out patch script */

/* Include variable and module definitions */
%include "&ppath.800_Var_List_v9.sas";    /* Defines variables, WOE, GRP and raw var names for modules */
%include "&ppath.802_Mod_List_v9.sas";    /* Defines raw module data, ENPUBLISH files and Multiout datasets */

/* Calculate date ranges for processing */
data _null_;                              /* Temporary data step for date calculations */
    call symput('str_dt_sas', put (intnx('month',today(),-3,'E'), date9.));    /* Start date: 3 months ago */
    call symput('end_dt_sas', put (intnx('month',today(),-1,'E'), date9.));    /* End date: 1 month ago */
run;

/* Set date variables */
%let str = &str_dt_sas.'d;    /* Format start date for SAS */
%let end = &end_dt_sas.'d;    /* Format end date for SAS */

/* Hardcoded dates (commented out) */
/*%let str = '31Jan2024'd;*/
/*%let end = '31Mar2024'd;*/

%put &str;    /* Debug print start date */
%put &end;    /* Debug print end date */

/*%let run_from_dt = '31Dec2023'd;*/    /* Commented out run from date */

/* Include scoring and processing scripts */
%include "&ppath.802_Score_launcher_v9.sas";     /* Main score launcher script */
%include "&ppath.802_Merge_Mod_v9.sas";          /* Model merging script */

/* Include alert processing scripts */
%include "&ppath.802_Tool_launcher_50bps_caus_v1_p_add.sas";    /* Alert tool launcher */

/* Launch Score Proc Means section */
%include "&ppath.700_Outputs_Proc_means.sas";    /* Process means calculations */
%include "&ppath.702_History_Append.sas";        /* Append history data */

/* Set governance paths and include related files */
%let gov = /sasdata/nacb/dev3/cb_production/100_arm_models/8002_Folder_Structure/03_Governance/code/;    /* Governance code path */
%include "&gov.ci_mod_bin_bmw.sas";             /* Include BMW model binning */
%include "&gov.ci_mod_bin_bctw.sas";            /* Include BCTW model binning */

/* Set model parameters */
%let model = CI;                /* Define model type */
%put &model;                    /* Debug print model type */

/* Set alert paths and include alert processing */
%let alerts = /sasdata/nacb/dev3/cb_production/200_bma_load/01_alert_reasons/;    /* Alert reasons path */
%include "alerts.02_Alert_Suppression_files.sas";    /* Include alert suppression logic */
%include "alerts.02_Alert_Suppression.sas";          /* Include additional alert suppression */

%include "&ppath.702_History_Append.sas";    /* Include history append logic again */
