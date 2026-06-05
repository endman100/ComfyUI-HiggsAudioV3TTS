import { app } from "../../scripts/app.js";

function isPlainWorkflowGraph(value) {
    return (
        value &&
        typeof value === "object" &&
        !Array.isArray(value) &&
        Array.isArray(value.nodes) &&
        Array.isArray(value.links) &&
        "last_node_id" in value
    );
}

function workflowNameFromFile(file) {
    return file.name.replace(/\.[^.]+$/, "");
}

function firstJsonFile(event) {
    const files = Array.from(event.dataTransfer?.files ?? []);
    return files.find((file) => file.name?.toLowerCase().endsWith(".json"));
}

app.registerExtension({
    name: "ComfyUI-HiggsAudioV3TTS.workflowDragDropCompat",

    async setup(appInstance) {
        if (appInstance.__higgsAudioV3WorkflowDragDropCompat) {
            return;
        }

        appInstance.__higgsAudioV3WorkflowDragDropCompat = true;
        const originalHandleFile = appInstance.handleFile?.bind(appInstance);
        if (!originalHandleFile) {
            return;
        }

        appInstance.handleFile = async function handleFileWithPlainWorkflowSupport(file, openSource, options) {
            if (file?.name?.toLowerCase().endsWith(".json")) {
                try {
                    const data = JSON.parse(await file.text());
                    if (isPlainWorkflowGraph(data)) {
                        await this.loadGraphData(data, true, true, workflowNameFromFile(file), {
                            openSource,
                            deferWarnings: options?.deferWarnings,
                        });
                        return true;
                    }
                } catch {
                    // Let ComfyUI's native importer report invalid JSON or unsupported files.
                }
            }

            return originalHandleFile(file, openSource, options);
        };

        document.addEventListener(
            "dragover",
            (event) => {
                if (firstJsonFile(event)) {
                    event.preventDefault();
                }
            },
            true
        );

        document.addEventListener(
            "drop",
            (event) => {
                const file = firstJsonFile(event);
                if (!file) {
                    return;
                }

                event.preventDefault();
                event.stopImmediatePropagation();
                void appInstance.handleFile(file, "drop");
            },
            true
        );
    },
});
