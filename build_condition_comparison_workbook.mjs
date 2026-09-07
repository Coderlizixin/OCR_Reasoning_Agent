import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = process.argv[2] ?? "test_case/RIG_Charxiv/evidence_conditions/condition_comparison_report.json";
const outputPath = process.argv[3] ?? "test_case/RIG_Charxiv/evidence_conditions/condition_comparison_report.xlsx";
const previewDir = process.argv[4] ?? "test_case/RIG_Charxiv/evidence_conditions/.report_build/previews";

const report = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();

const COLORS = {
  navy: "#17365D",
  teal: "#0F6B78",
  lightTeal: "#DDEBF7",
  pale: "#F3F7FA",
  gold: "#FFF2CC",
  green: "#E2F0D9",
  red: "#FCE4D6",
  white: "#FFFFFF",
  gray: "#666666",
};

const realConditions = report.sources.map((source) => source.condition_id);
const allConditions = [...realConditions, "random_mask_in_mean"];
const conditionMap = new Map(report.overall.map((row) => [row.condition_id, row]));
const sourceMap = new Map(report.sources.map((row) => [row.condition_id, row]));
const categoriesList = Object.entries(report.category_mapping).map(([categoryId, categoryName]) => ({
  category_id: Number(categoryId),
  category_name: categoryName,
}));

function excelColumn(indexOneBased) {
  let n = indexOneBased;
  let out = "";
  while (n > 0) {
    n -= 1;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

function styleTitle(sheet, rangeAddress) {
  const range = sheet.getRange(rangeAddress);
  range.format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 16 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  range.format.rowHeight = 28;
}

function styleHeader(range) {
  range.format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 30;
}

function styleSection(range) {
  range.format = {
    fill: COLORS.lightTeal,
    font: { bold: true, color: COLORS.navy, size: 12 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 22;
}

function configureSheet(sheet) {
  sheet.showGridlines = false;
}

function addTable(sheet, address, name, style = "TableStyleMedium2") {
  const table = sheet.tables.add(address, true, name);
  table.style = style;
  return table;
}

function sourceColumnMap() {
  const map = new Map();
  let col = 4;
  for (const conditionId of realConditions) {
    map.set(conditionId, {
      score: excelColumn(col),
      status: excelColumn(col + 1),
      finish: excelColumn(col + 2),
    });
    col += 3;
  }
  return map;
}

const perItemColumns = sourceColumnMap();
const perItemLastRow = report.per_item.length + 1;

// ---------------------------------------------------------------------------
// Per Item: the canonical paired data table used by the formula-driven sheets.
// ---------------------------------------------------------------------------
const perItem = workbook.worksheets.add("Per Item");
configureSheet(perItem);

const perItemHeaders = ["Index", "Category ID", "Category Name"];
for (const conditionId of realConditions) {
  perItemHeaders.push(`${conditionId} Score`, `${conditionId} Status`, `${conditionId} Finish Reason`);
}
perItemHeaders.push(
  "Random Mean Score",
  "GT In - Full",
  "Full - GT Out",
  "GT In - Random Mean",
  "GT ROI - GT In",
);

const perItemRows = report.per_item.map((item) => {
  const row = [item.index, item.inst_category, item.category_name];
  for (const conditionId of realConditions) {
    row.push(
      item[`${conditionId}_score`],
      item[`${conditionId}_status`],
      item[`${conditionId}_finish_reason`] ?? "",
    );
  }
  row.push(
    item.random_mask_in_mean_score,
    item.gt_in_minus_full,
    item.full_minus_gt_out,
    item.gt_in_minus_random_mean,
    item.gt_roi_minus_gt_in,
  );
  return row;
});

perItem.getRangeByIndexes(0, 0, 1, perItemHeaders.length).values = [perItemHeaders];
perItem.getRangeByIndexes(1, 0, perItemRows.length, perItemHeaders.length).values = perItemRows;
styleHeader(perItem.getRangeByIndexes(0, 0, 1, perItemHeaders.length));
addTable(perItem, `A1:${excelColumn(perItemHeaders.length)}${perItemLastRow}`, "PerItemTable", "TableStyleMedium2");
perItem.freezePanes.freezeRows(1);
perItem.freezePanes.freezeColumns(3);
perItem.getRange(`A2:B${perItemLastRow}`).format.horizontalAlignment = "center";
perItem.getRange(`D2:${excelColumn(perItemHeaders.length)}${perItemLastRow}`).format.horizontalAlignment = "center";
perItem.getRange(`Y2:Y${perItemLastRow}`).format.numberFormat = "0.000";
perItem.getRange(`Z2:AC${perItemLastRow}`).format.numberFormat = "0.000";
perItem.getRange("A:A").format.columnWidth = 9;
perItem.getRange("B:B").format.columnWidth = 11;
perItem.getRange("C:C").format.columnWidth = 21;
for (const columns of perItemColumns.values()) {
  perItem.getRange(`${columns.score}:${columns.score}`).format.columnWidth = 13;
  perItem.getRange(`${columns.status}:${columns.status}`).format.columnWidth = 14;
  perItem.getRange(`${columns.finish}:${columns.finish}`).format.columnWidth = 18;
}
perItem.getRange("Y:AC").format.columnWidth = 19;

// ---------------------------------------------------------------------------
// Summary: formula-driven overall metrics plus the pre-registered comparisons.
// ---------------------------------------------------------------------------
const summary = workbook.worksheets.add("Summary");
configureSheet(summary);
summary.mergeCells("A1:J1");
summary.getRange("A1").values = [["Evidence Condition Comparison Report"]];
styleTitle(summary, "A1:J1");
summary.getRange("A2:B5").values = [
  ["Model", report.model_name],
  ["Generated at", report.generated_at],
  ["Paired samples", report.sample_count],
  ["Bootstrap", `${report.bootstrap.samples} resamples; base seed ${report.bootstrap.base_seed}`],
];
summary.getRange("A2:A5").format.font = { bold: true, color: COLORS.navy };
summary.getRange("A2:B5").format.fill = COLORS.pale;
summary.getRange("B3").format.numberFormat = "yyyy-mm-dd hh:mm:ss";

summary.mergeCells("A6:J6");
summary.getRange("A6").values = [["Overall Results"]];
styleSection(summary.getRange("A6:J6"));

const overallHeaders = [
  "Condition ID", "Condition", "Correct", "Total", "Accuracy", "Valid Answers",
  "Valid Rate", "Truncated", "Truncation Rate", "Conditional Accuracy",
];
summary.getRange("A7:J7").values = [overallHeaders];
styleHeader(summary.getRange("A7:J7"));

const overallFirstRow = 8;
const rowByCondition = new Map();
allConditions.forEach((conditionId, offset) => rowByCondition.set(conditionId, overallFirstRow + offset));

for (const conditionId of realConditions) {
  const row = rowByCondition.get(conditionId);
  const source = sourceMap.get(conditionId);
  const columns = perItemColumns.get(conditionId);
  summary.getRange(`A${row}:B${row}`).values = [[conditionId, source.display_name]];
  summary.getRange(`C${row}:J${row}`).formulas = [[
    `=SUM('Per Item'!${columns.score}2:${columns.score}${perItemLastRow})`,
    `=COUNTA('Per Item'!A2:A${perItemLastRow})`,
    `=IFERROR(C${row}/D${row},0)`,
    `=COUNTIF('Per Item'!${columns.status}2:${columns.status}${perItemLastRow},"valid")`,
    `=IFERROR(F${row}/D${row},0)`,
    `=COUNTIF('Per Item'!${columns.status}2:${columns.status}${perItemLastRow},"truncated")`,
    `=IFERROR(H${row}/D${row},0)`,
    `=IFERROR(SUMIFS('Per Item'!${columns.score}2:${columns.score}${perItemLastRow},'Per Item'!${columns.status}2:${columns.status}${perItemLastRow},"valid")/F${row},0)`,
  ]];
}

const randomMeanRow = rowByCondition.get("random_mask_in_mean");
const randomSeedRows = realConditions
  .filter((conditionId) => conditionId.startsWith("random_mask_in_"))
  .map((conditionId) => rowByCondition.get(conditionId));
summary.getRange(`A${randomMeanRow}:B${randomMeanRow}`).values = [["random_mask_in_mean", "Random Mask-In (3-seed mean)"]];
summary.getRange(`C${randomMeanRow}:J${randomMeanRow}`).formulas = [[
  `=AVERAGE(${randomSeedRows.map((row) => `C${row}`).join(",")})`,
  `=D${randomSeedRows[0]}`,
  `=IFERROR(C${randomMeanRow}/D${randomMeanRow},0)`,
  `=AVERAGE(${randomSeedRows.map((row) => `F${row}`).join(",")})`,
  `=IFERROR(F${randomMeanRow}/D${randomMeanRow},0)`,
  `=AVERAGE(${randomSeedRows.map((row) => `H${row}`).join(",")})`,
  `=IFERROR(H${randomMeanRow}/D${randomMeanRow},0)`,
  `=IFERROR(SUM(${randomSeedRows.map((row) => `C${row}`).join(",")})/SUM(${randomSeedRows.map((row) => `F${row}`).join(",")}),0)`,
]];

const overallLastRow = overallFirstRow + allConditions.length - 1;
addTable(summary, `A7:J${overallLastRow}`, "OverallResultsTable", "TableStyleMedium2");
summary.getRange(`C8:D${overallLastRow}`).format.numberFormat = "0.00";
summary.getRange(`E8:E${overallLastRow}`).format.numberFormat = "0.00%";
summary.getRange(`F8:F${overallLastRow}`).format.numberFormat = "0.00";
summary.getRange(`G8:G${overallLastRow}`).format.numberFormat = "0.00%";
summary.getRange(`H8:H${overallLastRow}`).format.numberFormat = "0.00";
summary.getRange(`I8:J${overallLastRow}`).format.numberFormat = "0.00%";
summary.getRange(`A${randomMeanRow}:J${randomMeanRow}`).format.fill = COLORS.gold;

const primary = report.pairwise_comparisons;
const compareSectionRow = overallLastRow + 2;
summary.mergeCells(`A${compareSectionRow}:K${compareSectionRow}`);
summary.getRange(`A${compareSectionRow}`).values = [["Primary Paired Comparisons"]];
styleSection(summary.getRange(`A${compareSectionRow}:K${compareSectionRow}`));
const compareHeaderRow = compareSectionRow + 1;
const compareHeaders = [
  "Comparison", "Treatment", "Reference", "Delta (pp)", "95% CI Low", "95% CI High",
  "McNemar p", "Treatment Only", "Reference Only", "Both Correct", "Both Wrong",
];
summary.getRange(`A${compareHeaderRow}:K${compareHeaderRow}`).values = [compareHeaders];
styleHeader(summary.getRange(`A${compareHeaderRow}:K${compareHeaderRow}`));
const compareRows = primary.map((row) => [
  row.comparison_id,
  row.treatment_id,
  row.reference_id,
  row.delta_accuracy_pp,
  row.bootstrap_ci_95_pp[0],
  row.bootstrap_ci_95_pp[1],
  row.mcnemar_exact_p,
  row.treatment_only_indices.length,
  row.reference_only_indices.length,
  row.both_correct_indices.length,
  row.both_wrong_indices.length,
]);
summary.getRangeByIndexes(compareHeaderRow, 0, compareRows.length, compareHeaders.length).values = compareRows;
const compareLastRow = compareHeaderRow + compareRows.length;
addTable(summary, `A${compareHeaderRow}:K${compareLastRow}`, "PrimaryComparisonsTable", "TableStyleMedium4");
summary.getRange(`D${compareHeaderRow + 1}:F${compareLastRow}`).format.numberFormat = "0.000";
summary.getRange(`G${compareHeaderRow + 1}:G${compareLastRow}`).format.numberFormat = "0.000E+00";

const noteRow = compareLastRow + 2;
summary.mergeCells(`A${noteRow}:K${noteRow}`);
summary.getRange(`A${noteRow}`).values = [["Interpretation Notes"]];
styleSection(summary.getRange(`A${noteRow}:K${noteRow}`));
const notes = [
  "Accuracy denominators include all 41 questions; truncated outputs are counted as incorrect, per the experiment protocol.",
  "Paired confidence intervals use item-level bootstrap resampling. Random-mask aggregate intervals resample questions as clusters and retain all three seeds.",
  "McNemar exact p-values are reported only for binary one-condition-vs-one-condition comparisons.",
  "The source workbooks and every file listed in each FROZEN.json manifest were SHA-256 verified before analysis.",
];
notes.forEach((note, i) => {
  const row = noteRow + 1 + i;
  summary.mergeCells(`A${row}:K${row}`);
  summary.getRange(`A${row}`).values = [[note]];
  summary.getRange(`A${row}:K${row}`).format = { wrapText: true, font: { color: COLORS.gray, italic: true } };
  summary.getRange(`A${row}:K${row}`).format.rowHeight = 30;
});
summary.freezePanes.freezeRows(7);
summary.getRange("A:A").format.columnWidth = 29;
summary.getRange("B:B").format.columnWidth = 34;
summary.getRange("C:C").format.columnWidth = 24;
summary.getRange("D:J").format.columnWidth = 15;
summary.getRange("K:K").format.columnWidth = 13;

// ---------------------------------------------------------------------------
// Pairwise: full statistical details, including discordant question indices.
// ---------------------------------------------------------------------------
const pairwise = workbook.worksheets.add("Pairwise");
configureSheet(pairwise);
const pairHeaders = [
  "Comparison ID", "Treatment", "Reference", "Delta (pp)", "95% CI Low", "95% CI High",
  "McNemar p", "Treatment Only", "Reference Only", "Both Correct", "Both Wrong",
  "Treatment-only Indices", "Reference-only Indices", "Bootstrap Seed", "Bootstrap Samples", "CI Method",
];
const pairRows = report.pairwise_comparisons.map((row) => [
  row.comparison_id,
  row.treatment_id,
  row.reference_id,
  row.delta_accuracy_pp,
  row.bootstrap_ci_95_pp[0],
  row.bootstrap_ci_95_pp[1],
  row.mcnemar_exact_p ?? "",
  row.treatment_only_indices.length || "",
  row.reference_only_indices.length || "",
  row.both_correct_indices.length || "",
  row.both_wrong_indices.length || "",
  (row.treatment_only_indices ?? []).join(", "),
  (row.reference_only_indices ?? []).join(", "),
  row.bootstrap_seed,
  row.bootstrap_samples,
  row.paired_test,
]);
pairwise.getRangeByIndexes(0, 0, 1, pairHeaders.length).values = [pairHeaders];
pairwise.getRangeByIndexes(1, 0, pairRows.length, pairHeaders.length).values = pairRows;
styleHeader(pairwise.getRangeByIndexes(0, 0, 1, pairHeaders.length));
addTable(pairwise, `A1:${excelColumn(pairHeaders.length)}${pairRows.length + 1}`, "PairwiseTable", "TableStyleMedium4");
pairwise.freezePanes.freezeRows(1);
pairwise.getRange(`D2:F${pairRows.length + 1}`).format.numberFormat = "0.000";
pairwise.getRange(`G2:G${pairRows.length + 1}`).format.numberFormat = "0.000E+00";
pairwise.getRange("A:A").format.columnWidth = 34;
pairwise.getRange("B:C").format.columnWidth = 28;
pairwise.getRange("D:K").format.columnWidth = 15;
pairwise.getRange("L:M").format.columnWidth = 42;
pairwise.getRange("N:P").format.columnWidth = 18;
pairwise.getRange("A:P").format.wrapText = true;

// ---------------------------------------------------------------------------
// Categories: formula-driven category accuracy for every real/aggregate row.
// ---------------------------------------------------------------------------
const categories = workbook.worksheets.add("Categories");
configureSheet(categories);
const categoryHeaders = ["Condition ID", "Condition", "Category ID", "Category", "Correct", "Total", "Accuracy"];
categories.getRange("A1:G1").values = [categoryHeaders];
styleHeader(categories.getRange("A1:G1"));
let categoryRow = 2;
for (const conditionId of allConditions) {
  const display = conditionMap.get(conditionId)?.display_name ?? conditionId;
  for (const category of categoriesList) {
    categories.getRange(`A${categoryRow}:D${categoryRow}`).values = [[
      conditionId,
      display,
      category.category_id,
      category.category_name,
    ]];
    if (conditionId === "random_mask_in_mean") {
      const seedCategoryRows = randomSeedRows.map((_, idx) => {
        const seedId = realConditions.filter((id) => id.startsWith("random_mask_in_"))[idx];
        const sourceIndex = allConditions.indexOf(seedId);
        return 2 + sourceIndex * categoriesList.length + (category.category_id - 1);
      });
      categories.getRange(`E${categoryRow}:G${categoryRow}`).formulas = [[
        `=AVERAGE(${seedCategoryRows.map((row) => `E${row}`).join(",")})`,
        `=F${seedCategoryRows[0]}`,
        `=IFERROR(E${categoryRow}/F${categoryRow},0)`,
      ]];
    } else {
      const scoreCol = perItemColumns.get(conditionId).score;
      categories.getRange(`E${categoryRow}:G${categoryRow}`).formulas = [[
        `=SUMIFS('Per Item'!${scoreCol}2:${scoreCol}${perItemLastRow},'Per Item'!B2:B${perItemLastRow},C${categoryRow})`,
        `=COUNTIF('Per Item'!B2:B${perItemLastRow},C${categoryRow})`,
        `=IFERROR(E${categoryRow}/F${categoryRow},0)`,
      ]];
    }
    categoryRow += 1;
  }
}
const categoryLastRow = categoryRow - 1;
addTable(categories, `A1:G${categoryLastRow}`, "CategoryResultsTable", "TableStyleMedium2");
categories.freezePanes.freezeRows(1);
categories.getRange(`E2:F${categoryLastRow}`).format.numberFormat = "0.00";
categories.getRange(`G2:G${categoryLastRow}`).format.numberFormat = "0.00%";
categories.getRange("A:A").format.columnWidth = 30;
categories.getRange("B:B").format.columnWidth = 30;
categories.getRange("C:C").format.columnWidth = 12;
categories.getRange("D:D").format.columnWidth = 22;
categories.getRange("E:G").format.columnWidth = 14;

// ---------------------------------------------------------------------------
// Sources: provenance and frozen-manifest verification.
// ---------------------------------------------------------------------------
const sources = workbook.worksheets.add("Sources");
configureSheet(sources);
const sourceHeaders = [
  "Condition ID", "Condition", "Seed", "Run Directory", "Judge Workbook", "Workbook SHA-256",
  "Frozen At", "Frozen File Count", "Manifest SHA-256", "Verified",
];
const sourceRows = report.sources.map((row) => [
  row.condition_id,
  row.display_name,
  row.random_seed ?? "",
  row.run_directory,
  row.judge_workbook_path,
  row.judge_workbook_sha256,
  row.frozen_at,
  row.frozen_file_count,
  row.freeze_manifest_sha256,
  row.integrity_verified,
]);
sources.getRangeByIndexes(0, 0, 1, sourceHeaders.length).values = [sourceHeaders];
sources.getRangeByIndexes(1, 0, sourceRows.length, sourceHeaders.length).values = sourceRows;
styleHeader(sources.getRangeByIndexes(0, 0, 1, sourceHeaders.length));
addTable(sources, `A1:J${sourceRows.length + 1}`, "SourcesTable", "TableStyleMedium9");
sources.freezePanes.freezeRows(1);
sources.getRange("A:B").format.columnWidth = 30;
sources.getRange("C:C").format.columnWidth = 13;
sources.getRange("D:E").format.columnWidth = 62;
sources.getRange("F:F").format.columnWidth = 66;
sources.getRange("G:G").format.columnWidth = 24;
sources.getRange("H:H").format.columnWidth = 18;
sources.getRange("I:I").format.columnWidth = 66;
sources.getRange("J:J").format.columnWidth = 12;
sources.getRange("A:J").format.wrapText = true;
sources.getRange(`A2:J${sourceRows.length + 1}`).format.rowHeight = 48;
sources.getRange(`G2:G${sourceRows.length + 1}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";

// Conditional formatting on the most decision-relevant deltas.
summary.getRange(`D${compareHeaderRow + 1}:F${compareLastRow}`).conditionalFormats.addColorScale({
  minColor: "#F8696B",
  midColor: "#FFEB84",
  maxColor: "#63BE7B",
});
perItem.getRange(`Z2:AC${perItemLastRow}`).conditionalFormats.addColorScale({
  minColor: "#F8696B",
  midColor: "#FFEB84",
  maxColor: "#63BE7B",
});

// Run the required compact inspections and previews before the final export.
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const keyInspect = await workbook.inspect({
  kind: "table",
  range: `Summary!A1:K${noteRow + notes.length}`,
  include: "values,formulas",
  tableMaxRows: 40,
  tableMaxCols: 12,
});
const errorInspect = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});

for (const sheetName of ["Summary", "Pairwise", "Per Item", "Categories", "Sources"]) {
  const render = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await render.arrayBuffer()),
  );
}

const exportedWorkbook = await SpreadsheetFile.exportXlsx(workbook);
await exportedWorkbook.save(outputPath);

console.log(JSON.stringify({
  outputPath,
  previewDir,
  keyInspect,
  errorInspect,
}, null, 2));
