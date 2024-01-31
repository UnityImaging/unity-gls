import torch
import timm


class ClassModel(torch.nn.Module):
    def __init__(self, questions, question_names, question_weights, backbone_model='resnet50'):
        super().__init__()

        if backbone_model in ['coat_tiny', 'coat_lite_mini']:
            self.backbone = timm.create_model(backbone_model, num_classes=0, img_size=640)
        else:
            self.backbone = timm.create_model(backbone_model, num_classes=0)

        backbone_final_features_success = False

        if not backbone_final_features_success:
            try:
                self.backbone_final_features = timm.create_model(backbone_model).get_classifier().in_features
                backbone_final_features_success = True
            except:
                pass

        if not backbone_final_features_success:
            try:
                self.backbone_final_features = timm.create_model(backbone_model).head.in_features
                backbone_final_features_success = True
            except:
                pass

        if not backbone_final_features_success:
            raise Exception

        self.questions = questions
        self.question_names = question_names
        self.question_weights = question_weights
        self.classification_heads = []

        for question in questions:
            num_classes = len(question)
            head = torch.nn.Linear(in_features=self.backbone_final_features,
                                   out_features=num_classes)
            self.classification_heads.append(head)

        self.classification_heads = torch.nn.ModuleList(self.classification_heads)

    def forward(self, inputs):
        out = []
        x = self.backbone(inputs)
        for head in self.classification_heads:
            out.append(head(x))

        return out