import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = process.argv[2];
const outputPath = process.argv[3];
const previewDir = process.argv[4];
if (!inputPath || !outputPath || !previewDir) {
  throw new Error("usage: node build_cross_model_condition_workbook.mjs REPORT.json OUTPUT.xlsx PREVIEW_DIR");
}

const report = JSON.parse(await fs.readFile(inputPath, "utf8"));
const [smallModel, largeModel] = report.models;
const workbook = Workbook.create();

const COLORS = {
  navy: "#17365D",
  teal: "#0F6B78",
  pale: "#F3F7FA",
  white: "#FFFFFF",
  gold: "#FFF2CC",
};

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

function styleHeader(range) {
  range.format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 32;
}

function addTable(sheet, address, name, style = "TableStyleMedium2") {
  const table = sheet.tables.add(address, true, name);
  table.style = style;
  return table;
}

// ---------------------------------------------------------------------------
// Summary: formula-driven accuracies and cross-model deltas.
// ---------------------------------------------------------------------------
const summary = workbook.worksheets.add("Summary");
summary.mergeCells("A1:O1");
summary.getRange("A1").values = [["CharXiv Evidence Conditions — 9B / 27B Paired Summary"]];
summary.getRange("A1:O1").format = {
  fill: COLORS.navy,
  font: { bold: true, color: COLORS.white, size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:O1").format.rowHeight = 30;
summary.getRange("A2:O2").merge();
summary.getRange("A2").values = [[
  `Same ${report.sample_count} questions; 12,288-token cap; length-truncated outputs count as incorrect. Cross-model uncertainty uses question-level paired bootstrap.`,
]];
summary.getRange("A2:O2").format = { fill: COLORS.pale, font: { italic: true }, wrapText: true };
summary.getRange("A4:D6").values = [
  ["Field", "Value", "Field", "Value"],
  ["Smaller model", smallModel, "Larger model", largeModel],
  ["Generated (UTC)", report.generated_at, "Frozen sources verified", report.input_reports.every((row) => row.all_sources_frozen_and_verified)],
];
styleHeader(summary.getRange("A4:D4"));
summary.getRange("B6").format.numberFormat = "yyyy-mm-dd hh:mm:ss";

const summaryHeaders = [
  "Condition",
  `${smallModel} correct`,
  `${smallModel} total`,
  `${smallModel} accuracy`,
  `${largeModel} correct`,
  `${largeModel} total`,
  `${largeModel} accuracy`,
  "27B - 9B (pp)",
  "95% CI low (pp)",
  "95% CI high (pp)",
  "Paired p",
  `${smallModel} valid rate`,
  `${largeModel} valid rate`,
  `${smallModel} truncated`,
  `${largeModel} truncated`,
];
summary.getRange("A8:O8").values = [summaryHeaders];
styleHeader(summary.getRange("A8:O8"));
report.overall_by_condition.forEach((row, offset) => {
  const excelRow = 9 + offset;
  const small = row.models[smallModel];
  const large = row.models[largeModel];
  summary.getRange(`A${excelRow}:C${excelRow}`).values = [[row.display_name, small.correct, small.total]];
  summary.getRange(`D${excelRow}`).formulas = [[`=IFERROR(B${excelRow}/C${excelRow},0)`]];
  summary.getRange(`E${excelRow}:F${excelRow}`).values = [[large.correct, large.total]];
  summary.getRange(`G${excelRow}:H${excelRow}`).formulas = [[
    `=IFERROR(E${excelRow}/F${excelRow},0)`,
    `=(G${excelRow}-D${excelRow})*100`,
  ]];
  summary.getRange(`I${excelRow}:O${excelRow}`).values = [[
    row.bootstrap_ci_95_pp[0],
    row.bootstrap_ci_95_pp[1],
    row.p_value,
    small.valid_answer_rate,
    large.valid_answer_rate,
    small.truncated,
    large.truncated,
  ]];
});
const summaryLastRow = 8 + report.overall_by_condition.length;
addTable(summary, `A8:O${summaryLastRow}`, "CrossModelSummaryTable", "TableStyleMedium2");
summary.freezePanes.freezeRows(8);
summary.getRange(`D9:D${summaryLastRow}`).format.numberFormat = "0.00%";
summary.getRange(`G9:G${summaryLastRow}`).format.numberFormat = "0.00%";
summary.getRange(`H9:J${summaryLastRow}`).format.numberFormat = "0.00";
summary.getRange(`K9:K${summaryLastRow}`).format.numberFormat = "0.000E+00";
summary.getRange(`L9:M${summaryLastRow}`).format.numberFormat = "0.00%";
summary.getRange(`H9:H${summaryLastRow}`).conditionalFormats.addColorScale({
  minColor: "#F8696B",
  midColor: "#FFEB84",
  maxColor: "#63BE7B",
});
summary.getRange("A:A").format.columnWidth = 34;
summary.getRange("B:O").format.columnWidth = 16;
summary.getRange("A:O").format.wrapText = true;

// ---------------------------------------------------------------------------
// H1 effects: within-model effects and the paired difference-in-differences.
// ---------------------------------------------------------------------------
const effects = workbook.worksheets.add("H1 Effects");
const effectHeaders = [
  "Comparison",
  "Treatment",
  "Reference",
  "9B delta (pp)",
  "9B CI low",
  "9B CI high",
  "9B paired p",
  "27B delta (pp)",
  "27B CI low",
  "27B CI high",
  "27B paired p",
  "Interaction (pp)",
  "Interaction CI low",
  "Interaction CI high",
];
effects.getRange("A1:N1").values = [effectHeaders];
styleHeader(effects.getRange("A1:N1"));
report.hypothesis_effects.forEach((row, offset) => {
  const excelRow = 2 + offset;
  const small = row.models[smallModel];
  const large = row.models[largeModel];
  effects.getRange(`A${excelRow}:K${excelRow}`).values = [[
    row.comparison_id,
    row.treatment_id,
    row.reference_id,
    small.delta_accuracy_pp,
    small.bootstrap_ci_95_pp[0],
    small.bootstrap_ci_95_pp[1],
    small.mcnemar_exact_p,
    large.delta_accuracy_pp,
    large.bootstrap_ci_95_pp[0],
    large.bootstrap_ci_95_pp[1],
    large.mcnemar_exact_p,
  ]];
  effects.getRange(`L${excelRow}`).formulas = [[`=H${excelRow}-D${excelRow}`]];
  effects.getRange(`M${excelRow}:N${excelRow}`).values = [[
    row.bootstrap_ci_95_pp[0],
    row.bootstrap_ci_95_pp[1],
  ]];
});
const effectLastRow = report.hypothesis_effects.length + 1;
addTable(effects, `A1:N${effectLastRow}`, "CrossModelEffectsTable", "TableStyleMedium4");
effects.freezePanes.freezeRows(1);
effects.getRange(`D2:F${effectLastRow}`).format.numberFormat = "0.00";
effects.getRange(`H2:J${effectLastRow}`).format.numberFormat = "0.00";
effects.getRange(`L2:N${effectLastRow}`).format.numberFormat = "0.00";
effects.getRange(`G2:G${effectLastRow}`).format.numberFormat = "0.000E+00";
effects.getRange(`K2:K${effectLastRow}`).format.numberFormat = "0.000E+00";
effects.getRange(`L2:L${effectLastRow}`).conditionalFormats.addColorScale({
  minColor: "#F8696B",
  midColor: "#FFEB84",
  maxColor: "#63BE7B",
});
effects.getRange("A:C").format.columnWidth = 28;
effects.getRange("D:N").format.columnWidth = 16;
effects.getRange("A:N").format.wrapText = true;

// ---------------------------------------------------------------------------
// Category statistics.
// ---------------------------------------------------------------------------
const categories = workbook.worksheets.add("Categories");
const categoryHeaders = ["Model", "Condition", "Category ID", "Category", "Correct", "Total", "Accuracy"];
categories.getRange("A1:G1").values = [categoryHeaders];
styleHeader(categories.getRange("A1:G1"));
const categoryRows = report.category_statistics.map((row) => [
  row.model_name,
  row.display_name,
  row.inst_category,
  row.category_name,
  row.correct,
  row.total,
  row.accuracy,
]);
categories.getRange(`A2:G${categoryRows.length + 1}`).values = categoryRows;
addTable(categories, `A1:G${categoryRows.length + 1}`, "CrossModelCategoriesTable", "TableStyleMedium2");
categories.freezePanes.freezeRows(1);
categories.getRange(`G2:G${categoryRows.length + 1}`).format.numberFormat = "0.00%";
categories.getRange("A:B").format.columnWidth = 31;
categories.getRange("C:C").format.columnWidth = 13;
categories.getRange("D:D").format.columnWidth = 22;
categories.getRange("E:G").format.columnWidth = 14;

// ---------------------------------------------------------------------------
// Per-item audit table, one 9B/27B/delta triplet per condition.
// ---------------------------------------------------------------------------
const perItem = workbook.worksheets.add("Per Item");
const conditionIds = report.overall_by_condition.map((row) => row.condition_id);
const displayById = new Map(report.overall_by_condition.map((row) => [row.condition_id, row.display_name]));
const perItemHeaders = ["Index", "Category ID", "Category"];
for (const conditionId of conditionIds) {
  const display = displayById.get(conditionId);
  perItemHeaders.push(`${display} 9B`, `${display} 27B`, `${display} delta`);
}
perItem.getRangeByIndexes(0, 0, 1, perItemHeaders.length).values = [perItemHeaders];
styleHeader(perItem.getRangeByIndexes(0, 0, 1, perItemHeaders.length));
report.per_item.forEach((item, offset) => {
  const excelRow = 2 + offset;
  perItem.getRange(`A${excelRow}:C${excelRow}`).values = [[item.index, item.inst_category, item.category_name]];
  let column = 4;
  for (const conditionId of conditionIds) {
    const condition = item.conditions[conditionId];
    perItem.getRangeByIndexes(excelRow - 1, column - 1, 1, 2).values = [[
      condition[smallModel].score,
      condition[largeModel].score,
    ]];
    const smallColumn = excelColumn(column);
    const largeColumn = excelColumn(column + 1);
    perItem.getRangeByIndexes(excelRow - 1, column + 1, 1, 1).formulas = [[
      `=${largeColumn}${excelRow}-${smallColumn}${excelRow}`,
    ]];
    column += 3;
  }
});
const perItemLastRow = report.per_item.length + 1;
const perItemLastColumn = excelColumn(perItemHeaders.length);
addTable(perItem, `A1:${perItemLastColumn}${perItemLastRow}`, "CrossModelPerItemTable", "TableStyleMedium9");
perItem.freezePanes.freezeRows(1);
perItem.freezePanes.freezeColumns(3);
perItem.getRange("A:A").format.columnWidth = 9;
perItem.getRange("B:B").format.columnWidth = 12;
perItem.getRange("C:C").format.columnWidth = 22;
perItem.getRange(`D:${perItemLastColumn}`).format.columnWidth = 15;
for (let column = 6; column <= perItemHeaders.length; column += 3) {
  const letter = excelColumn(column);
  perItem.getRange(`${letter}2:${letter}${perItemLastRow}`).conditionalFormats.addColorScale({
    minColor: "#F8696B",
    midColor: "#FFEB84",
    maxColor: "#63BE7B",
  });
}

// ---------------------------------------------------------------------------
// Sources and hashes.
// ---------------------------------------------------------------------------
const sources = workbook.worksheets.add("Sources");
sources.getRange("A1:G1").values = [["Type", "Model", "Condition", "Path", "SHA-256", "Files", "Verified"]];
styleHeader(sources.getRange("A1:G1"));
const sourceRows = [];
for (const input of report.input_reports) {
  sourceRows.push(["condition report", input.model_name, "all", input.path, input.sha256, input.frozen_file_count, input.all_sources_frozen_and_verified]);
}
for (const model of report.models) {
  for (const source of report.verified_frozen_sources[model]) {
    sourceRows.push([
      "frozen run",
      model,
      source.condition_id,
      source.freeze_manifest_path,
      source.freeze_manifest_sha256,
      source.frozen_file_count,
      source.integrity_verified,
    ]);
  }
}
sources.getRange(`A2:G${sourceRows.length + 1}`).values = sourceRows;
addTable(sources, `A1:G${sourceRows.length + 1}`, "CrossModelSourcesTable", "TableStyleMedium9");
sources.freezePanes.freezeRows(1);
sources.getRange("A:C").format.columnWidth = 26;
sources.getRange("D:D").format.columnWidth = 72;
sources.getRange("E:E").format.columnWidth = 68;
sources.getRange("F:G").format.columnWidth = 13;
sources.getRange("A:G").format.wrapText = true;
sources.getRange(`A2:G${sourceRows.length + 1}`).format.rowHeight = 48;

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const keyInspect = await workbook.inspect({
  kind: "table",
  range: `Summary!A1:O${summaryLastRow}`,
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 15,
});
const errorInspect = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});

for (const sheetName of ["Summary", "H1 Effects", "Categories", "Per Item", "Sources"]) {
  const render = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await render.arrayBuffer()));
}

const exportedWorkbook = await SpreadsheetFile.exportXlsx(workbook);
await exportedWorkbook.save(outputPath);

console.log(JSON.stringify({ outputPath, previewDir, keyInspect, errorInspect }, null, 2));
