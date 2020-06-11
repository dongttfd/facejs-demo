const maxHeight = 768
const maxWith = 1024
const video = document.getElementById('video')
const photoInput = document.getElementById('photoInput')

const start = async () => {
    document.body.append('Loaded')

    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)

    // training
    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, .6)

    let image, canvas
    photoInput.addEventListener('change', async () => {

        if (image) {
            image.remove()
        }

        if (canvas) {
            canvas.remove()
        }

        image = await faceapi.bufferToImage(photoInput.files[0])
        container.append(image)

        canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)

        const displaySize = getDisplaySize(image)

        canvas.width = displaySize.width
        canvas.height = displaySize.height
        image.width = displaySize.width
        image.height = displaySize.height

        faceapi.matchDimensions(canvas, displaySize)

        const detections = await faceapi.detectAllFaces(image)
            .withFaceLandmarks()
            .withFaceDescriptors()
            .withFaceExpressions()

        const resizedDetections = faceapi.resizeResults(detections, displaySize)

        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))

        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box

            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })

            drawBox.draw(canvas)
        })
    })
}

const loadLabeledImages = () => {
    const names = ['dongtt', 'nhung']

    return Promise.all(
        names.map(async name => {
            const descriptions = []
            for (let i = 1; i < 3; i++) {

                const image = await faceapi.fetchImage(`http://localhost:8888//photos-train/${name}/${i}.JPG`)
                const detections = await faceapi.detectSingleFace(image)
                    .withFaceLandmarks()
                    .withFaceDescriptor()

                console.log(i, name, detections)

                descriptions.push(detections.descriptor)
            }

            return new faceapi.LabeledFaceDescriptors(name, descriptions)
        })
    )
}

const getDisplaySize = (image) => {
    let width = image.width
    let height = image.height

    if (width >= height) {
        height = (maxWith * height) / width
        width = maxWith
    } else {
        width = (maxHeight * width) / height
        height = maxHeight
    }

    return { width, height }
}

const startVideo = async () => {
    // training
    const videoPreview = document.getElementById('videoPreview')
    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, .6)

    navigator.getUserMedia(
        { video: {} },
        stream => video.srcObject = stream,
        error => console.error(error)
    )


    video.addEventListener('play', () => {
        const canvas = faceapi.createCanvasFromMedia(video)

        videoPreview.append(canvas)

        const displaySize = getDisplaySize(video)

        canvas.width = displaySize.width
        canvas.height = displaySize.height
        video.width = displaySize.width
        video.height = displaySize.height

        faceapi.matchDimensions(canvas, displaySize)

        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                .withFaceLandmarks()
                .withFaceExpressions()
                .withFaceDescriptors()
                .withAgeAndGender()

            const resizedDetections = faceapi.resizeResults(detections, displaySize)

            const results = resizedDetections.map(d => {
                const result = faceMatcher.findBestMatch(d.descriptor)
                result.age = d.age
                result.gender = d.gender

                return result
            })

            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)

            results.forEach((result, i) => {
                const box = resizedDetections[i].detection.box
                const drawBox = new faceapi.draw.DrawBox(
                    box,
                    { label: `${result.toString()} / gender: ${result.gender}/ age: ${result.age.toString().split('.')[0]} ` }
                )

                drawBox.draw(canvas)

                faceapi.draw.drawFaceLandmarks(canvas, resizedDetections[i])
                faceapi.draw.drawFaceExpressions(canvas, resizedDetections[i])
            })
        }, 100)
    })
}


const interpolateAgePredictions = (age) => {
    predictAges = [age].concat()
}


Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
    faceapi.nets.ageGenderNet.loadFromUri('/models')

    // ]).then(start)
]).then(startVideo)
